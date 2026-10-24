//! Fused Metal kernels for the cv3 DiT hot path, hosted through candle
//! custom ops (`CustomOp3::metal_fwd`), so the fusion needs no candle fork
//! dispatch: everything below runs against candle's public API
//! (`MetalDevice` encoder/builder access, `MetalStorage::new`,
//! `apply_op3_no_bwd`).
//!
//! Kernel-design rules learned the hard way, kept here for the next kernel:
//! * Bind kernel-written buffers with `Output::new` — a plain input-style
//!   binding skips candle's hazard tracking, so the readback races the
//!   kernel and reads zeros.
//! * Never decompose `tid` with runtime `%`/`/` by non-powers-of-two: GPU
//!   integer division is an emulated ~30-cycle sequence per thread and
//!   dominates simple kernels (measured 4x). Use a 3-D grid
//!   `(channels, seq, batch)` with `threads_per_grid` instead.
//! * Outputs written as planes of one buffer (`(2,B,T,D)`) surface as
//!   zero-copy `narrow(0, i, 1)` views for downstream ops.
//!
//! Numeric contracts (gated in `flow::dit::tests`):
//! * `rope_headsplit` is bit-identical to the production chain
//!   (slice-copy/rope_i/cat/transpose): the rotation expression order
//!   matches `rope_i`'s kernel, passthrough channels are copies.
//! * `gate_residual_modln`/`gate_residual_add` differ from the op chain only
//!   by Metal's fma contraction of `x + g*attn` (one rounding instead of
//!   two) and the one-pass (`E[x^2] - mean^2`) vs Welford variance
//!   (~2e-7 relative) — both pure float rounding, gated at 1e-6/1e-5.

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CustomOp3, DType, Device, Layout, MetalDevice, MetalStorage, Result, Shape, Tensor, bail,
};
use candle_metal_kernels::Output;
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use candle_metal_kernels::utils::set_param;
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const SRC: &str = include_str!("metal_fused.metal");

const ROPE_HS: &str = "cv3_rope_headsplit_f32";
const GATE_MODLN: &str = "cv3_gate_residual_modln_f32";
const GATE_TAIL: &str = "cv3_gate_tail_modln_f32";
/// The gate kernel hardcodes 1024-wide threadgroup scratch (== the cv3 DiT
/// model dim, a power of two).
const GATE_MODLN_DIM: usize = 1024;

/// Eligibility for the complete boundary-normalization chain. Unsupported
/// models must stay on the reference path before any carry is produced.
pub(crate) fn supports_gate(x: &Tensor) -> bool {
    x.device().is_metal()
        && x.dtype() == DType::F32
        && x.dims().last() == Some(&GATE_MODLN_DIM)
        && x.is_contiguous()
}

/// Compile-once cache of the fused kernels, keyed by (name, device).
type PipelineCache = Mutex<HashMap<(&'static str, DeviceId), Arc<ComputePipeline>>>;

fn pipeline(dev: &MetalDevice, name: &'static str) -> Result<Arc<ComputePipeline>> {
    static CACHE: OnceLock<PipelineCache> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (name, dev.id());
    if let Ok(guard) = cache.lock()
        && let Some(p) = guard.get(&key)
    {
        return Ok(p.clone());
    }
    let lib = dev
        .metal_device()
        .new_library_with_source(SRC, None)
        .map_err(|e| candle_core::Error::msg(format!("cv3 fused lib compile: {e}")))?;
    let fun = lib
        .get_function(name, None)
        .map_err(|e| candle_core::Error::msg(format!("cv3 fused fn {name}: {e}")))?;
    let pl = dev
        .new_compute_pipeline_state_with_function(&fun)
        .map_err(|e| candle_core::Error::msg(format!("cv3 fused pipeline {name}: {e}")))?;
    let pl = Arc::new(pl);
    if let Ok(mut guard) = cache.lock() {
        guard.insert(key, pl.clone());
    }
    Ok(pl)
}

fn buf_param(enc: &ComputeCommandEncoder, position: usize, s: &MetalStorage, l: &Layout) {
    set_param(
        enc,
        position,
        (s.buffer(), l.start_offset() * s.dtype().size_in_bytes()),
    );
}

fn size3(w: usize, h: usize, d: usize) -> MTLSize {
    MTLSize {
        width: w,
        height: h,
        depth: d,
    }
}

struct RopeHeadSplitOp {
    dev: MetalDevice,
    heads: usize,
    head_dim: usize,
}

impl CustomOp3 for RopeHeadSplitOp {
    fn name(&self) -> &'static str {
        "cv3-rope-headsplit"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        bail!("cv3 fused rope/head-split is Metal-only")
    }

    fn metal_fwd(
        &self,
        qs: &MetalStorage,
        ql: &Layout,
        cs: &MetalStorage,
        cl: &Layout,
        ss: &MetalStorage,
        sl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (b, t, inner) = ql.shape().dims3()?;
        let el = b * self.heads * t * self.head_dim;
        let out = self
            .dev
            .new_buffer_builder()
            .with_size_for(el, DType::F32)
            .with_label("cv3-rope-hs")
            .build()?;
        let pl = pipeline(&self.dev, ROPE_HS)?;
        let guard = self.dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(&pl);
        set_param(enc, 0, self.head_dim);
        set_param(enc, 1, self.head_dim.trailing_zeros() as usize);
        buf_param(enc, 2, qs, ql);
        buf_param(enc, 3, cs, cl);
        buf_param(enc, 4, ss, sl);
        set_param(enc, 5, Output::new(out.as_ref()));
        enc.dispatch_thread_groups(size3(inner.div_ceil(256), t, b), size3(256, 1, 1));
        drop(guard);
        Ok((
            MetalStorage::new(out, self.dev.clone(), el, DType::F32),
            (b, self.heads, t, self.head_dim).into(),
        ))
    }
}

/// Fused partial-rope + head-split. `q`: `[B,T,inner]` contiguous F32 with
/// `inner == heads*head_dim` and `head_dim` a power of two; `cos`/`sin`:
/// `[T, head_dim/2]` contiguous. Returns `[B,H,T,D]`, bit-identical to the
/// production slice-copy/rope_i/cat/transpose chain.
pub(crate) fn rope_headsplit(
    q: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let (b, t, inner) = q.dims3()?;
    if inner != heads * head_dim
        || head_dim < 2
        || inner % 256 != 0
        || head_dim.count_ones() != 1
        || q.dtype() != DType::F32
        || !q.is_contiguous()
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || cos.dims() != [t, head_dim / 2]
        || sin.dims() != [t, head_dim / 2]
        || !cos.is_contiguous()
        || !sin.is_contiguous()
    {
        bail!("rope_headsplit: unsupported shape/layout {:?}", q.shape());
    }
    let dev = match q.device() {
        Device::Metal(d) => d,
        _ => bail!("rope_headsplit: metal device required"),
    };
    let _ = b;
    q.apply_op3_no_bwd(
        cos,
        sin,
        &RopeHeadSplitOp {
            dev: dev.clone(),
            heads,
            head_dim,
        },
    )
}

struct GateResidualModLnOp {
    dev: MetalDevice,
}

impl CustomOp3 for GateResidualModLnOp {
    fn name(&self) -> &'static str {
        "cv3-gate-residual-modln"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        bail!("cv3 fused gate/mod-LN is Metal-only")
    }

    fn metal_fwd(
        &self,
        attn_s: &MetalStorage,
        attn_l: &Layout,
        x_s: &MetalStorage,
        x_l: &Layout,
        emb_s: &MetalStorage,
        emb_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (b, t, dim) = attn_l.shape().dims3()?;
        let rows = b * t;
        let el = 2 * rows * dim;
        let out = self
            .dev
            .new_buffer_builder()
            .with_size_for(el, DType::F32)
            .with_label("cv3-gate-modln")
            .build()?;
        let pl = pipeline(&self.dev, GATE_MODLN)?;
        let guard = self.dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(&pl);
        set_param(enc, 0, dim);
        buf_param(enc, 1, attn_s, attn_l);
        buf_param(enc, 2, x_s, x_l);
        buf_param(enc, 3, emb_s, emb_l);
        set_param(enc, 4, Output::new(out.as_ref()));
        // One thread group per (b,t) row, D threads per group.
        enc.dispatch_thread_groups(size3(1, t, b), size3(GATE_MODLN_DIM, 1, 1));
        drop(guard);
        Ok((
            MetalStorage::new(out, self.dev.clone(), el, DType::F32),
            (2, b, t, dim).into(),
        ))
    }
}

/// Fused gate region producing `(2,B,T,D)`: plane 0 is
/// `modLN(x + gate_msa⊙attn, scale_mlp+1, shift_mlp)` (the FFN input),
/// plane 1 is the post-gate residual `x + gate_msa⊙attn` (the base for the
/// block tail). `attn`/`x`: `[B,T,D]` contiguous F32 with `D == 1024`;
/// `emb`: the raw `[B,1,6D]` AdaLayerNorm embedding (modulation read by
/// channel offset, no packing kernel).
pub(crate) fn gate_residual_modln(attn: &Tensor, x: &Tensor, emb: &Tensor) -> Result<Tensor> {
    let (b, t, dim) = attn.dims3()?;
    if dim != GATE_MODLN_DIM
        || attn.dtype() != DType::F32
        || !attn.is_contiguous()
        || x.dtype() != DType::F32
        || emb.dtype() != DType::F32
        || x.dims() != [b, t, dim]
        || !x.is_contiguous()
        || emb.dims() != [b, 1, 6 * dim]
        || !emb.is_contiguous()
    {
        bail!("gate_residual_modln: unsupported shape/layout");
    }
    let dev = match attn.device() {
        Device::Metal(d) => d,
        _ => bail!("gate_residual_modln: metal device required"),
    };
    attn.apply_op3_no_bwd(x, emb, &GateResidualModLnOp { dev: dev.clone() })
}

struct GateTailModLnOp {
    dev: MetalDevice,
}

impl CustomOp3 for GateTailModLnOp {
    fn name(&self) -> &'static str {
        "cv3-gate-tail-modln"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        bail!("cv3 fused gate-tail/mod-LN is Metal-only")
    }

    fn metal_fwd(
        &self,
        ff_s: &MetalStorage,
        ff_l: &Layout,
        z_s: &MetalStorage,
        z_l: &Layout,
        pk_s: &MetalStorage,
        pk_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (b, t, dim) = ff_l.shape().dims3()?;
        let rows = b * t;
        let el = 2 * rows * dim;
        let out = self
            .dev
            .new_buffer_builder()
            .with_size_for(el, DType::F32)
            .with_label("cv3-gate-tail")
            .build()?;
        let pl = pipeline(&self.dev, GATE_TAIL)?;
        let guard = self.dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(&pl);
        set_param(enc, 0, dim);
        buf_param(enc, 1, ff_s, ff_l);
        buf_param(enc, 2, z_s, z_l);
        buf_param(enc, 3, pk_s, pk_l);
        set_param(enc, 4, Output::new(out.as_ref()));
        enc.dispatch_thread_groups(size3(1, t, b), size3(GATE_MODLN_DIM, 1, 1));
        drop(guard);
        Ok((
            MetalStorage::new(out, self.dev.clone(), el, DType::F32),
            (2, b, t, dim).into(),
        ))
    }
}

/// Fused block tail + boundary layer norm producing `(2,B,T,D)`: plane 0 is
/// `z + gate_mlp⊙ff` (the residual carry), plane 1 is
/// `modLN(plane0, scale_next+1, shift_next)` (the next block's attention
/// input, or the final norm feeding `proj_out`). `ff`/`z`: `[B,T,D]`
/// contiguous F32; `packed`: `[B,2,6D]` contiguous
/// `cat([emb_this, emb_next], 1)` with `emb_next` in AdaLayerNorm channel
/// order (shift at 0D, scale at D); for the final block pass the reordered
/// final-norm pseudo-embedding.
pub(crate) fn gate_tail_modln(ff: &Tensor, z: &Tensor, packed: &Tensor) -> Result<Tensor> {
    let (b, t, dim) = ff.dims3()?;
    if dim != GATE_MODLN_DIM
        || ff.dtype() != DType::F32
        || !ff.is_contiguous()
        || z.dtype() != DType::F32
        || packed.dtype() != DType::F32
        || z.dims() != [b, t, dim]
        || !z.is_contiguous()
        || packed.dims() != [b, 2, 6 * dim]
        || !packed.is_contiguous()
    {
        bail!("gate_tail_modln: unsupported shape/layout");
    }
    let dev = match ff.device() {
        Device::Metal(d) => d,
        _ => bail!("gate_tail_modln: metal device required"),
    };
    ff.apply_op3_no_bwd(z, packed, &GateTailModLnOp { dev: dev.clone() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Metal device; no models"]
    fn rejects_mixed_dtype_rope_tables() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let q = Tensor::zeros((1, 2, 1024), DType::F32, &dev)?;
        let table = Tensor::ones((2, 32), DType::F16, &dev)?;
        assert!(rope_headsplit(&q, &table, &table, 16, 64).is_err());
        Ok(())
    }
}
