#include <metal_stdlib>
using namespace metal;

// Fused qkv split + interleaved rope + head-major transpose. Grid
// (3*hidden, T, B); qkv (B,T,3*hidden) contiguous with hidden =
// heads*head_dim (both powers of two, so every decomposition below is a
// shift/mask); cos/sin (T, head_dim/2) contiguous; out
// (3,B,heads,T,head_dim) contiguous with planes 0=q roped, 1=k roped,
// 2=v copy. Plane = pos.x >> hidden_shift selects the qkv third; the
// rotation expressions match candle's `ropei` kernel verbatim so the
// result is bit-identical to split+rope_i (the whole-model head_dim is
// 64 for the shipped checkpoint, so all heads rotate).
kernel void itb_qkv_split_rope_f32(
    constant size_t &hidden,
    constant size_t &hidden_shift,
    constant size_t &head_dim,
    constant size_t &head_shift,
    device const float *qkv,
    device const float *cos_t,
    device const float *sin_t,
    device float *out,
    uint3 pos [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]
) {
  const size_t plane = pos.x >> hidden_shift;
  const size_t local = pos.x & (hidden - 1);
  const size_t d = local & (head_dim - 1);
  const size_t h = local >> head_shift;
  const size_t heads = hidden >> head_shift;
  const size_t src_i = (pos.z * grid_dim.y + pos.y) * (3 * hidden) + pos.x;
  const size_t dst_i =
      (((plane * grid_dim.z + pos.z) * heads + h) * grid_dim.y + pos.y) * head_dim + d;
  if (plane == 2) {
    out[dst_i] = qkv[src_i];
    return;
  }
  const size_t half_dim = head_dim / 2;
  const size_t i = d >> 1;
  const float c = cos_t[pos.y * half_dim + i];
  const float s = sin_t[pos.y * half_dim + i];
  if ((d & 1) == 0) {
    out[dst_i] = qkv[src_i] * c - qkv[src_i + 1] * s;
  } else {
    out[dst_i] = qkv[src_i - 1] * s + qkv[src_i] * c;
  }
}

// Gather side of a strided transposed convolution (the cheap half of the
// GEMM+gather factorization; the GEMM half runs as a plain candle matmul on
// contiguous operands). After col = x^T @ reshape(w, (c_in, c_out*k)) the
// output is out[c_out, l_out] = sum over the taps k_x == (out_x + pad) mod
// stride + j*stride of col[(out_x + pad - k_x)/stride, c_out, k_x]. With
// stride a power of two dividing k, every output reads exactly k/stride
// col elements and all arithmetic is shift/mask. Grid (l_out, c_out, B).
kernel void itb_convt_gather_f32(
    constant size_t &l_out,
    constant size_t &l_in,
    constant size_t &c_out,
    constant size_t &k,
    constant size_t &stride,
    constant size_t &stride_shift,
    constant size_t &pad,
    device const float *col, // (B, l_in, c_out, k) contiguous
    device float *out,       // (B, c_out, l_out) contiguous
    uint3 pos [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]
) {
  const size_t b = pos.z;
  const size_t oc = pos.y;
  const size_t ox = pos.x;
  // The x grid is rounded up to whole 256-thread groups.
  if (ox >= l_out) {
    return;
  }
  const size_t phase = (ox + pad) & (stride - 1);
  const size_t base = (ox + pad - phase) >> stride_shift;
  // tap k_x = phase + j*stride reads col[base - j]; j counts down from the
  // first tap <= ox + pad with a valid input index.
  float d = 0.0;
  for (size_t kk = phase; kk < k; kk += stride) {
    const size_t j = (kk - phase) >> stride_shift;
    const size_t i = base - j;
    if (i < l_in) {
      d += col[((b * l_in + i) * c_out + oc) * k + kk];
    }
  }
  out[(b * c_out + oc) * l_out + ox] = d;
}

// Fused up-FIR + snake activation for BigVGAN's alias-free activation:
// up[2m] = Σ_j odd_taps[j]·xp[7+m-j], up[2m+1] = Σ_j even_taps[j]·xp[8+m-j]
// with xp the replicate-padded input (index clamp, no materialized pad) and
// taps pre-scaled by AA_RATIO; then y = u + scale_c·sin²(a_c·u) on the way
// out. Replaces the pad copy, two phase convolutions and the five-kernel
// snake elementwise chain with one dispatch. Grid (2T, C, B).
kernel void itb_fir_up_snake_f32(
    constant size_t &t_len,
    constant size_t &c_len,
    constant size_t &half_k,
    device const float *x,        // (B, C, T)
    device const float *phases,   // (2, half_k): [odd_fwd; even_fwd]
    device const float *acts,     // (2, C): [a; scale]
    device float *out,            // (B, C, 2T) snaked
    uint3 pos [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]
) {
  const size_t ox = pos.x;
  if (ox >= 2 * t_len) {
    return;
  }
  const size_t c = pos.y;
  const size_t b = pos.z;
  const size_t m = ox >> 1;
  // up[2m] is the odd phase (crop start 7), up[2m+1] the even (start 8);
  // replicate pad is 5 on each side for the (AA_KERNEL=12, AA_RATIO=2)
  // hyperparameters this kernel is gated to.
  const bool odd = (ox & 1) == 0;
  const size_t start = odd ? 7 : 8;
  const size_t phase_row = odd ? 0 : 1;
  const device const float *taps = phases + phase_row * half_k;
  const size_t xbase = (b * c_len + c) * t_len;
  float d = 0.0;
  for (size_t j = 0; j < half_k; ++j) {
    long xi = (long)(start + m) - (long)j - 5;
    xi = clamp(xi, 0L, (long)t_len - 1);
    d += taps[j] * x[xbase + (size_t)xi];
  }
  const float a = acts[c];
  const float sc = acts[c_len + c];
  const float sn = sin(a * d);
  out[(b * c_len + c) * 2 * t_len + ox] = d + sc * (sn * sn);
}

// Wavenet gated activation: out[(b,h,t)] = tanh(v0)·sigmoid(v1) with
// v0 = x_in[b,h,t] + g[b,g_off+h], v1 = x_in[b,H+h,t] + g[b,g_off+H+h].
// tanh/sigmoid are candle's expressions verbatim (precise::tanh,
// recip(1+exp(-x))) so the result is bit-identical to the six-kernel
// chain. Grid (T, H, B); x_in (B,2H,T) contiguous, g (B,g_len,1)
// contiguous, out (B,H,T) contiguous.
kernel void itb_wn_gate_act_f32(
    constant size_t &t_len,
    constant size_t &hidden,
    constant size_t &g_len,
    constant size_t &g_off,
    device const float *x_in,
    device const float *g,
    device float *out,
    uint3 pos [[thread_position_in_grid]]
) {
  const size_t t = pos.x;
  if (t >= t_len) {
    return;
  }
  const size_t h = pos.y;
  const size_t b = pos.z;
  const size_t two_h = 2 * hidden;
  const float v0 = x_in[(b * two_h + h) * t_len + t] + g[b * g_len + g_off + h];
  const float v1 =
      x_in[(b * two_h + hidden + h) * t_len + t] + g[b * g_len + g_off + hidden + h];
  out[(b * hidden + h) * t_len + t] =
      precise::tanh(v0) * (1.0 / (1.0 + exp(-v1)));
}

// Wavenet res/skip tail: rs (B,2H,T), x/output (B,H,T) all contiguous.
// Plane 0 = x + rs[:, :H] (next layer input), plane 1 = output + rs[:, H:]
// (skip accumulator) of a (2,B,H,T) buffer. Grid (T, H, B).
kernel void itb_wn_res_skip_tail_f32(
    constant size_t &t_len,
    constant size_t &hidden,
    constant size_t &b_len,
    device const float *rs,
    device const float *x,
    device const float *output,
    device float *out,
    uint3 pos [[thread_position_in_grid]]
) {
  const size_t t = pos.x;
  if (t >= t_len) {
    return;
  }
  const size_t h = pos.y;
  const size_t b = pos.z;
  const size_t plane = b * hidden * t_len + h * t_len + t;
  const size_t two_h = 2 * hidden;
  out[plane] = x[plane] + rs[(b * two_h + h) * t_len + t];
  out[b_len * hidden * t_len + plane] =
      output[plane] + rs[(b * two_h + hidden + h) * t_len + t];
}
