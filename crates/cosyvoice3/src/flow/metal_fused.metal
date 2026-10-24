#include <metal_stdlib>
using namespace metal;

// Partial interleaved rope + head-split. Grid (inner, seq, B); q (B,T,inner)
// contiguous with inner = heads*head_dim; cos/sin (T, head_dim/2) contiguous
// unique tables; out (B,H,T,D) contiguous. Only head 0 (flat channel <
// head_dim) rotates. head_dim is a power of two, so every decomposition is a
// shift/mask.
kernel void cv3_rope_headsplit_f32(
    constant size_t &head_dim,
    constant size_t &head_shift,
    device const float *q,
    device const float *cos_t,
    device const float *sin_t,
    device float *out,
    uint3 pos [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]
) {
  const size_t flat = pos.x;
  const size_t d = flat & (head_dim - 1);
  const size_t h = flat >> head_shift;
  const size_t heads = grid_dim.x >> head_shift;
  const size_t src_i = (pos.z * grid_dim.y + pos.y) * grid_dim.x + flat;
  const size_t dst_i = ((pos.z * heads + h) * grid_dim.y + pos.y) * head_dim + d;
  if (h != 0) {
    out[dst_i] = q[src_i];
    return;
  }
  const size_t half_dim = head_dim / 2;
  const size_t i = d >> 1;
  const float c = cos_t[pos.y * half_dim + i];
  const float s = sin_t[pos.y * half_dim + i];
  if ((d & 1) == 0) {
    out[dst_i] = q[src_i] * c - q[src_i + 1] * s;
  } else {
    out[dst_i] = q[src_i - 1] * s + q[src_i] * c;
  }
}

// Fused gate region, one (b,t) row per threadgroup. Grid (D, T, B),
// threadgroup width D. With z = x + gate_msa*attn writes TWO planes of a
// (2,B,T,D) output: plane 0 = modLN(z, scale_mlp+1, shift_mlp) (the FFN
// input), plane 1 = z (the residual base for the block tail). attn/x:
// (B,T,D) contiguous; emb: (B,1,6D) contiguous raw AdaLayerNorm embedding
// (gate_msa at 2D, shift_mlp at 3D, scale_mlp at 4D). Requires D == 1024
// (threadgroup arrays, power of two).
kernel void cv3_gate_residual_modln_f32(
    constant size_t &dim,
    device const float *attn,
    device const float *x,
    device const float *emb,
    device float *out,
    uint3 tpos [[thread_position_in_threadgroup]],
    uint3 gpos [[threadgroup_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]
) {
  threadgroup float tz[1024];
  threadgroup float tz2[1024];
  const size_t j = tpos.x;
  const size_t b = gpos.z;
  const size_t row = b * grid_dim.y + gpos.y;
  const size_t base = row * dim;
  const size_t erow = b * 6 * dim + j;
  const float g = emb[erow + 2 * dim];
  const float sc = emb[erow + 4 * dim] + 1.0;
  const float sh = emb[erow + 3 * dim];
  const float z = x[base + j] + g * attn[base + j];
  tz[j] = z;
  tz2[j] = z * z;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (size_t stride = dim / 2; stride > 0; stride >>= 1) {
    if (j < stride) {
      tz[j] += tz[j + stride];
      tz2[j] += tz2[j + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float mean = tz[0] / (float)dim;
  const float var = tz2[0] / (float)dim - mean * mean;
  const float rstd = rsqrt(var + 1e-6);
  const float normalized = (z - mean) * rstd;
  out[base + j] = fma(normalized, sc, sh);
  out[grid_dim.y * grid_dim.z * dim + base + j] = z;
}

// Block tail + boundary layer norm, one (b,t) row per threadgroup. Grid
// (D, T, B), threadgroup width D. With v = z + gate_mlp*ff writes TWO
// planes of a (2,B,T,D) output: plane 0 = v (the residual carry), plane 1 =
// modLN(v, scale_next+1, shift_next) — the next block's attention input
// (or the final norm feeding proj_out). ff/z: (B,T,D) contiguous; packed:
// (B,2,6D) contiguous concatenation [this block's raw AdaLayerNorm
// embedding (gate_mlp at 5D), next modulator in AdaLayerNorm channel order
// (shift at 0D, scale at D)]. Requires D == 1024 (threadgroup arrays).
kernel void cv3_gate_tail_modln_f32(
    constant size_t &dim,
    device const float *ff,
    device const float *z,
    device const float *packed,
    device float *out,
    uint3 tpos [[thread_position_in_threadgroup]],
    uint3 gpos [[threadgroup_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]
) {
  threadgroup float tz[1024];
  threadgroup float tz2[1024];
  const size_t j = tpos.x;
  const size_t b = gpos.z;
  const size_t row = b * grid_dim.y + gpos.y;
  const size_t base = row * dim;
  const size_t p0 = b * 12 * dim;
  const float g2 = packed[p0 + 5 * dim + j];
  const float sh_n = packed[p0 + 6 * dim + j];
  const float sc_n = packed[p0 + 7 * dim + j] + 1.0;
  const float v = z[base + j] + g2 * ff[base + j];
  tz[j] = v;
  tz2[j] = v * v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (size_t stride = dim / 2; stride > 0; stride >>= 1) {
    if (j < stride) {
      tz[j] += tz[j + stride];
      tz2[j] += tz2[j + stride];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const float mean = tz[0] / (float)dim;
  const float var = tz2[0] / (float)dim - mean * mean;
  const float rstd = rsqrt(var + 1e-6);
  out[base + j] = v;
  out[grid_dim.y * grid_dim.z * dim + base + j] =
      fma((v - mean) * rstd, sc_n, sh_n);
}
