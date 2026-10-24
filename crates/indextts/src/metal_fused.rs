//! Fused Metal kernels for the IndexTTS hot path, hosted through candle
//! custom ops (`CustomOp3::metal_fwd`) so the fusion needs no candle fork:
//! everything runs against candle's public API (`MetalDevice`
//! encoder/builder access, `MetalStorage::new`, `apply_op3_no_bwd`).
//!
//! Kernel-design rules inherited from the cv3 template
//! (`crates/cosyvoice3/src/flow/metal_fused.rs`):
//! * Bind kernel-written buffers with `Output::new` — a plain input-style
//!   binding skips candle's hazard tracking, so readback races the kernel.
//! * Never decompose `tid` with runtime `%`/`/` by non-powers-of-two; use a
//!   3-D grid with `threads_per_grid` and shift/mask arithmetic only.
//! * Outputs written as planes of one buffer surface as zero-copy
//!   `narrow(0, i, 1)` views for downstream ops (candle's Metal sdpa passes
//!   `start_offset`/`stride` explicitly, so offset views are accepted).
//!
//! Numeric contract (gated in `dit::tests`): `qkv_split_rope` is
//! bit-identical to the production split/`rope_i` chain — the rotation
//! expressions match candle's `ropei` kernel verbatim and the v plane is a
//! pure copy.

use candle_core::backend::BackendStorage;
use candle_core::metal_backend::DeviceId;
use candle_core::{
    CustomOp1, CustomOp2, CustomOp3, DType, Device, Layout, MetalDevice, MetalStorage, Result,
    Shape, Tensor, bail,
};
use candle_metal_kernels::Output;
use candle_metal_kernels::metal::{ComputeCommandEncoder, ComputePipeline};
use candle_metal_kernels::utils::set_param;
use objc2_metal::MTLSize;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

const SRC: &str = include_str!("metal_fused.metal");

const QKV_SPLIT_ROPE: &str = "itb_qkv_split_rope_f32";
const CONVT_GATHER: &str = "itb_convt_gather_f32";
const FIR_UP_SNAKE: &str = "itb_fir_up_snake_f32";
const WN_GATE_ACT: &str = "itb_wn_gate_act_f32";
const WN_RES_SKIP_TAIL: &str = "itb_wn_res_skip_tail_f32";

/// im2col1d through candle's own kernel (bit-identical to the materialized
/// buffer inside candle's Metal conv1d), exposed so the host can pair it
/// with a contiguous-operand GEMM that writes the (c_out, l_out) layout
/// directly — skipping candle's strided-B batched matmul and the
/// transpose-back copy.
struct Im2col1dOp {
    dev: MetalDevice,
    k: usize,
    padding: usize,
    dilation: usize,
}

impl CustomOp1 for Im2col1dOp {
    fn name(&self) -> &'static str {
        "itb-im2col1d"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        bail!("itb im2col1d is Metal-only")
    }

    fn metal_fwd(&self, xs: &MetalStorage, xl: &Layout) -> Result<(MetalStorage, Shape)> {
        let (b, c_in, l_in) = xl.shape().dims3()?;
        let l_out = (l_in + 2 * self.padding - self.dilation * (self.k - 1) - 1) + 1;
        let dst_el = b * l_out * c_in * self.k;
        let out = self
            .dev
            .new_buffer_builder()
            .with_size_for(dst_el, DType::F32)
            .with_label("itb-im2col1d")
            .build()?;
        let guard = self.dev.command_encoder()?;
        candle_metal_kernels::call_im2col1d_strided(
            self.dev.metal_device(),
            &guard,
            self.dev.kernels(),
            "im2col1d_f32",
            xl.shape().dims(),
            xl.stride(),
            (self.k, 1, self.padding, self.dilation),
            candle_metal_kernels::BufferOffset {
                buffer: xs.buffer(),
                offset_in_bytes: xl.start_offset() * DType::F32.size_in_bytes(),
            },
            &out,
        )
        .map_err(candle_core::Error::wrap)?;
        drop(guard);
        Ok((
            MetalStorage::new(out, self.dev.clone(), dst_el, DType::F32),
            (b, l_out, c_in * self.k).into(),
        ))
    }
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
        .map_err(|e| candle_core::Error::msg(format!("itb fused lib compile: {e}")))?;
    let fun = lib
        .get_function(name, None)
        .map_err(|e| candle_core::Error::msg(format!("itb fused fn {name}: {e}")))?;
    let pl = dev
        .new_compute_pipeline_state_with_function(&fun)
        .map_err(|e| candle_core::Error::msg(format!("itb fused pipeline {name}: {e}")))?;
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

struct QkvSplitRopeOp {
    dev: MetalDevice,
    heads: usize,
    head_dim: usize,
}

impl CustomOp3 for QkvSplitRopeOp {
    fn name(&self) -> &'static str {
        "itb-qkv-split-rope"
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
        bail!("itb fused qkv split/rope is Metal-only")
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
        let (b, t, three_hidden) = ql.shape().dims3()?;
        let hidden = self.heads * self.head_dim;
        let el = 3 * b * self.heads * t * self.head_dim;
        let out = self
            .dev
            .new_buffer_builder()
            .with_size_for(el, DType::F32)
            .with_label("itb-qkv-split-rope")
            .build()?;
        let pl = pipeline(&self.dev, QKV_SPLIT_ROPE)?;
        let guard = self.dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(&pl);
        set_param(enc, 0, hidden);
        set_param(enc, 1, hidden.trailing_zeros() as usize);
        set_param(enc, 2, self.head_dim);
        set_param(enc, 3, self.head_dim.trailing_zeros() as usize);
        buf_param(enc, 4, qs, ql);
        buf_param(enc, 5, cs, cl);
        buf_param(enc, 6, ss, sl);
        set_param(enc, 7, Output::new(out.as_ref()));
        debug_assert_eq!(three_hidden, 3 * hidden);
        enc.dispatch_thread_groups(size3(three_hidden.div_ceil(256), t, b), size3(256, 1, 1));
        drop(guard);
        Ok((
            MetalStorage::new(out, self.dev.clone(), el, DType::F32),
            (3, b, self.heads, t, self.head_dim).into(),
        ))
    }
}

/// Fused qkv split + interleaved rope + head-major transpose. `qkv`:
/// `[B,T,3*hidden]` contiguous F32 with `hidden == heads*head_dim` and both
/// `hidden`/`head_dim` powers of two; `cos`/`sin`: `[T, head_dim/2]`
/// contiguous. Returns `[3,B,H,T,D]` (q roped, k roped, v copy),
/// bit-identical to the production split/`rope_i` chain; the three planes
/// are consumed as zero-copy `narrow(0, i, 1)` views.
pub(crate) fn qkv_split_rope(
    qkv: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    let hidden = heads * head_dim;
    let (b, t, three_hidden) = qkv.dims3()?;
    if three_hidden != 3 * hidden
        || hidden.count_ones() != 1
        || head_dim < 2
        || three_hidden % 256 != 0
        || head_dim.count_ones() != 1
        || qkv.dtype() != DType::F32
        || !qkv.is_contiguous()
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || cos.dims() != [t, head_dim / 2]
        || sin.dims() != [t, head_dim / 2]
        || !cos.is_contiguous()
        || !sin.is_contiguous()
    {
        bail!("qkv_split_rope: unsupported shape/layout {:?}", qkv.shape());
    }
    let dev = match qkv.device() {
        Device::Metal(d) => d,
        _ => bail!("qkv_split_rope: metal device required"),
    };
    let _ = b;
    qkv.apply_op3_no_bwd(
        cos,
        sin,
        &QkvSplitRopeOp {
            dev: dev.clone(),
            heads,
            head_dim,
        },
    )
}

struct ConvtGatherOp {
    dev: MetalDevice,
    c_out: usize,
    k: usize,
    l_out: usize,
    stride: usize,
    pad: usize,
}

impl CustomOp1 for ConvtGatherOp {
    fn name(&self) -> &'static str {
        "itb-convt-gather"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        bail!("itb convt gather is Metal-only")
    }

    fn metal_fwd(&self, cs: &MetalStorage, cl: &Layout) -> Result<(MetalStorage, Shape)> {
        let (b, l_in, _) = cl.shape().dims3()?;
        let el = b * self.c_out * self.l_out;
        let out = self
            .dev
            .new_buffer_builder()
            .with_size_for(el, DType::F32)
            .with_label("itb-convt-gather")
            .build()?;
        let pl = pipeline(&self.dev, CONVT_GATHER)?;
        let guard = self.dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(&pl);
        set_param(enc, 0, self.l_out);
        set_param(enc, 1, l_in);
        set_param(enc, 2, self.c_out);
        set_param(enc, 3, self.k);
        set_param(enc, 4, self.stride);
        set_param(enc, 5, self.stride.trailing_zeros() as usize);
        set_param(enc, 6, self.pad);
        buf_param(enc, 7, cs, cl);
        set_param(enc, 8, Output::new(out.as_ref()));
        enc.dispatch_thread_groups(
            size3(self.l_out.div_ceil(256), self.c_out, b),
            size3(256, 1, 1),
        );
        drop(guard);
        Ok((
            MetalStorage::new(out, self.dev.clone(), el, DType::F32),
            (b, self.c_out, self.l_out).into(),
        ))
    }
}

/// Fused strided transposed convolution as GEMM + gather: with `stride`
/// dividing `k` (both powers of two here), `x^T @ reshape(w, (c_in, c_out*k))`
/// followed by one gather kernel per output element reproduces
/// `ConvTranspose1d` exactly (same tap order; the c_in reduction order is
/// the matmul's, gated at 1e-5 relative against the reference path). The
/// bias add runs after the gather exactly like candle's conv path
/// (elementwise, bit-determined).
/// `x`: `[B,c_in,l_in]`, `w`: `[c_in,c_out,k]`, `bias`: `[c_out]` or None;
/// returns `[B,c_out,l_out]` with `l_out = (l_in-1)*stride - 2*padding + k`.
pub(crate) fn conv_transpose1d_fused(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    padding: usize,
) -> Result<Tensor> {
    let (b, c_in, l_in) = x.dims3()?;
    let (cin2, c_out, k) = w.dims3()?;
    if cin2 != c_in
        || stride < 2
        || k % stride != 0
        || stride.count_ones() != 1
        || k / stride > 4
        || x.dtype() != DType::F32
        || !x.is_contiguous()
        || !w.is_contiguous()
    {
        bail!("conv_transpose1d_fused: unsupported shape/layout");
    }
    let l_out = l_in
        .checked_sub(1)
        .and_then(|n| n.checked_mul(stride))
        .and_then(|n| n.checked_add(k))
        .and_then(|n| padding.checked_mul(2).and_then(|p| n.checked_sub(p)))
        .filter(|&n| n > 0)
        .ok_or_else(|| candle_core::Error::msg("conv_transpose1d_fused: invalid output length"))?;
    let dev = match x.device() {
        Device::Metal(d) => d,
        _ => bail!("conv_transpose1d_fused: metal device required"),
    };
    // GEMM half on contiguous operands: (B,l_in,c_in) @ (1,c_in,c_out*k).
    let xt = x.transpose(1, 2)?.contiguous()?;
    let wr = w.reshape((c_in, c_out * k))?.unsqueeze(0)?;
    let col = xt.matmul(&wr)?; // (B,l_in,c_out*k)
    let _ = b;
    let out = col.apply_op1_no_bwd(&ConvtGatherOp {
        dev: dev.clone(),
        c_out,
        k,
        l_out,
        stride,
        pad: padding,
    })?;
    match bias {
        Some(bias) if bias.dims() == [c_out] => out.broadcast_add(&bias.reshape((1, c_out, 1))?),
        Some(_) => bail!("conv_transpose1d_fused: bias must be [c_out]"),
        None => Ok(out),
    }
}

/// Fused stride-1 conv1d as im2col + direct-layout GEMM: candle's Metal
/// conv1d materializes the same im2col buffer but then runs a batched
/// matmul against a strided transposed-kernel view and copies the result
/// back into channels-first layout. This variant keeps candle's im2col
/// kernel (bit-identical) and issues one batched GEMM
/// `reshape(w,(1,c_out,c_in*k)) @ col^T` whose output already is
/// `(B, c_out, l_out)`. The K-dim reduction order is the matmul's tile
/// order, gated at 1e-5 relative against candle's `Conv1d`. The bias add
/// runs after the GEMM exactly like candle's conv path (elementwise, so
/// bit-determined).
/// `x`: `[B,c_in,l_in]`, `w`: `[c_out,c_in,k]`, `bias`: `[c_out]` or None.
pub(crate) fn conv1d_gemmdirect(
    x: &Tensor,
    w: &Tensor,
    bias: Option<&Tensor>,
    padding: usize,
    dilation: usize,
) -> Result<Tensor> {
    let (_, c_in, l_in) = x.dims3()?;
    let (c_out, cin2, k) = w.dims3()?;
    if cin2 != c_in
        || x.dtype() != DType::F32
        || w.dtype() != DType::F32
        || !x.is_contiguous()
        || !w.is_contiguous()
    {
        bail!("conv1d_gemmdirect: unsupported shape/layout");
    }
    if l_in + 2 * padding < dilation * (k - 1) + 1 {
        bail!("conv1d_gemmdirect: empty output l_out");
    }
    let dev = match x.device() {
        Device::Metal(d) => d,
        _ => bail!("conv1d_gemmdirect: metal device required"),
    };
    let col = x.apply_op1_no_bwd(&Im2col1dOp {
        dev: dev.clone(),
        k,
        padding,
        dilation,
    })?; // (B, l_out, c_in*k)
    let w2 = w.reshape((1, c_out, c_in * k))?;
    let out = w2.matmul(&col.transpose(1, 2)?)?; // (B, c_out, l_out)
    match bias {
        Some(bias) if bias.dims() == [c_out] => out.broadcast_add(&bias.reshape((1, c_out, 1))?),
        Some(_) => bail!("conv1d_gemmdirect: bias must be [c_out]"),
        None => Ok(out),
    }
}

/// Wavenet gated activation `tanh(v)·sigmoid(v+H)` with
/// `v = x_in + g` (upstream `fused_add_tanh_sigmoid_multiply`): replaces the
/// broadcast add + two strided narrows + tanh + sigmoid + multiply (six
/// kernels and three full intermediates) with one dispatch. The tanh and
/// sigmoid expressions are candle's verbatim (`precise::tanh`,
/// `recip(1+exp(-x))`), so the result is bit-identical.
/// `x_in`: `[B,2H,T]` contiguous F32; `g`: the `[B,2H*n,T_g]` contiguous
/// conditioning with `T_g == 1`, read at channel offset `g_off`.
pub(crate) fn wn_gate_act(x_in: &Tensor, g: &Tensor, g_off: usize) -> Result<Tensor> {
    let (b, two_h, _t) = x_in.dims3()?;
    let (gb, g_len, gt) = g.dims3()?;
    if two_h % 2 != 0
        || gb != b
        || gt != 1
        || g_off + two_h > g_len
        || x_in.dtype() != DType::F32
        || g.dtype() != DType::F32
        || !x_in.is_contiguous()
        || !g.is_contiguous()
    {
        bail!("wn_gate_act: unsupported shape/layout");
    }
    let dev = match x_in.device() {
        Device::Metal(d) => d,
        _ => bail!("wn_gate_act: metal device required"),
    };
    let hidden = two_h / 2;
    x_in.apply_op2_no_bwd(
        g,
        &WnGateActOp {
            dev: dev.clone(),
            hidden,
            g_off,
        },
    )
}

/// Wavenet res/skip tail for non-final layers, as one dual-plane kernel:
/// plane 0 = `x + rs[:, :H]` (the next layer's input), plane 1 =
/// `output + rs[:, H:]` (the skip accumulator). Replaces two strided
/// narrows and two adds (four kernels, one full intermediate) with one
/// dispatch returning `[2,B,H,T]` consumed as zero-copy `narrow` planes.
pub(crate) fn wn_res_skip_tail(rs: &Tensor, x: &Tensor, output: &Tensor) -> Result<Tensor> {
    let (b, two_h, t) = rs.dims3()?;
    let hidden = two_h / 2;
    if two_h % 2 != 0
        || x.dims() != [b, hidden, t]
        || x.dtype() != DType::F32
        || output.dtype() != DType::F32
        || output.dims() != [b, hidden, t]
        || rs.dtype() != DType::F32
        || !rs.is_contiguous()
        || !x.is_contiguous()
        || !output.is_contiguous()
    {
        bail!("wn_res_skip_tail: unsupported shape/layout");
    }
    let dev = match rs.device() {
        Device::Metal(d) => d,
        _ => bail!("wn_res_skip_tail: metal device required"),
    };
    rs.apply_op3_no_bwd(
        x,
        output,
        &WnResSkipTailOp {
            dev: dev.clone(),
            hidden,
        },
    )
}

struct FirUpSnakeOp {
    dev: MetalDevice,
    half_k: usize,
}

impl CustomOp3 for FirUpSnakeOp {
    fn name(&self) -> &'static str {
        "itb-fir-up-snake"
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
        bail!("itb fused fir-up/snake is Metal-only")
    }

    fn metal_fwd(
        &self,
        xs: &MetalStorage,
        xl: &Layout,
        ps: &MetalStorage,
        pl: &Layout,
        as_: &MetalStorage,
        al: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (b, c, t) = xl.shape().dims3()?;
        let el = b * c * 2 * t;
        let out = self
            .dev
            .new_buffer_builder()
            .with_size_for(el, DType::F32)
            .with_label("itb-fir-up-snake")
            .build()?;
        let pl_ = pipeline(&self.dev, FIR_UP_SNAKE)?;
        let guard = self.dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(&pl_);
        set_param(enc, 0, t);
        set_param(enc, 1, c);
        set_param(enc, 2, self.half_k);
        buf_param(enc, 3, xs, xl);
        buf_param(enc, 4, ps, pl);
        buf_param(enc, 5, as_, al);
        set_param(enc, 6, Output::new(out.as_ref()));
        enc.dispatch_thread_groups(size3((2 * t).div_ceil(256), c, b), size3(256, 1, 1));
        drop(guard);
        Ok((
            MetalStorage::new(out, self.dev.clone(), el, DType::F32),
            (b, c, 2 * t).into(),
        ))
    }
}

/// Fused up-FIR + snake: one dispatch replacing the materialized replicate
/// pad, the two phase convolutions and the snake elementwise chain.
/// `x`: `[B,C,T]` contiguous F32; `phases`: `[2,half_k]` contiguous
/// (row 0 odd taps, row 1 even taps, AA_RATIO-scaled, forward order);
/// `acts`: `[2,C]` contiguous (row 0 `a`, row 1 `1/(beta+1e-9)`). Returns
/// the snaked `[B,C,2T]` upsample. Numeric gate at 1e-5 relative against
/// the `fir_upsample` + `apply_activation` chain.
pub(crate) fn fir_up_snake(
    x: &Tensor,
    phases: &Tensor,
    acts: &Tensor,
    half_k: usize,
) -> Result<Tensor> {
    let (b, c, t) = x.dims3()?;
    if x.dtype() != DType::F32
        || !x.is_contiguous()
        || phases.dtype() != DType::F32
        || acts.dtype() != DType::F32
        || phases.dims() != [2, half_k]
        || !phases.is_contiguous()
        || acts.dims() != [2, c]
        || !acts.is_contiguous()
        || t == 0
    {
        bail!("fir_up_snake: unsupported shape/layout");
    }
    let dev = match x.device() {
        Device::Metal(d) => d,
        _ => bail!("fir_up_snake: metal device required"),
    };
    let _ = b;
    x.apply_op3_no_bwd(
        phases,
        acts,
        &FirUpSnakeOp {
            dev: dev.clone(),
            half_k,
        },
    )
}

struct WnGateActOp {
    dev: MetalDevice,
    hidden: usize,
    g_off: usize,
}

impl CustomOp2 for WnGateActOp {
    fn name(&self) -> &'static str {
        "itb-wn-gate-act"
    }

    fn cpu_fwd(
        &self,
        _: &candle_core::CpuStorage,
        _: &Layout,
        _: &candle_core::CpuStorage,
        _: &Layout,
    ) -> Result<(candle_core::CpuStorage, Shape)> {
        bail!("itb wavenet gate-act is Metal-only")
    }

    fn metal_fwd(
        &self,
        xs: &MetalStorage,
        xl: &Layout,
        gs: &MetalStorage,
        gl: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (b, _two_h, t) = xl.shape().dims3()?;
        let g_len = gl.shape().dims3()?.1;
        let el = b * self.hidden * t;
        let out = self
            .dev
            .new_buffer_builder()
            .with_size_for(el, DType::F32)
            .with_label("itb-wn-gate-act")
            .build()?;
        let pl = pipeline(&self.dev, WN_GATE_ACT)?;
        let guard = self.dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(&pl);
        set_param(enc, 0, t);
        set_param(enc, 1, self.hidden);
        set_param(enc, 2, g_len);
        set_param(enc, 3, self.g_off);
        buf_param(enc, 4, xs, xl);
        buf_param(enc, 5, gs, gl);
        set_param(enc, 6, Output::new(out.as_ref()));
        enc.dispatch_thread_groups(size3(t.div_ceil(256), self.hidden, b), size3(256, 1, 1));
        drop(guard);
        Ok((
            MetalStorage::new(out, self.dev.clone(), el, DType::F32),
            (b, self.hidden, t).into(),
        ))
    }
}

struct WnResSkipTailOp {
    dev: MetalDevice,
    hidden: usize,
}

impl CustomOp3 for WnResSkipTailOp {
    fn name(&self) -> &'static str {
        "itb-wn-res-skip-tail"
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
        bail!("itb wavenet res/skip tail is Metal-only")
    }

    fn metal_fwd(
        &self,
        rs_s: &MetalStorage,
        rs_l: &Layout,
        x_s: &MetalStorage,
        x_l: &Layout,
        o_s: &MetalStorage,
        o_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        let (b, _, t) = rs_l.shape().dims3()?;
        let el = 2 * b * self.hidden * t;
        let out = self
            .dev
            .new_buffer_builder()
            .with_size_for(el, DType::F32)
            .with_label("itb-wn-res-skip-tail")
            .build()?;
        let pl = pipeline(&self.dev, WN_RES_SKIP_TAIL)?;
        let guard = self.dev.command_encoder()?;
        let enc: &ComputeCommandEncoder = guard.as_ref();
        enc.set_compute_pipeline_state(&pl);
        set_param(enc, 0, t);
        set_param(enc, 1, self.hidden);
        set_param(enc, 2, b);
        buf_param(enc, 3, rs_s, rs_l);
        buf_param(enc, 4, x_s, x_l);
        buf_param(enc, 5, o_s, o_l);
        set_param(enc, 6, Output::new(out.as_ref()));
        enc.dispatch_thread_groups(size3(t.div_ceil(256), self.hidden, b), size3(256, 1, 1));
        drop(guard);
        Ok((
            MetalStorage::new(out, self.dev.clone(), el, DType::F32),
            (2, b, self.hidden, t).into(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a Metal device; no models"]
    fn contiguous_batch_slice_preserves_conv_offset() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let cpu = Tensor::arange(0f32, 64f32, &Device::Cpu)?.reshape((2, 2, 16))?;
        let weights = Tensor::ones((3, 2, 3), DType::F32, &Device::Cpu)?;
        let expected = cpu.narrow(0, 1, 1)?.conv1d(&weights, 1, 1, 1, 1)?;
        let base = cpu.to_device(&dev)?;
        let slice = base.narrow(0, 1, 1)?;
        assert!(slice.is_contiguous());
        let actual = conv1d_gemmdirect(&slice, &weights.to_device(&dev)?, None, 1, 1)?;
        assert_eq!(
            actual.flatten_all()?.to_vec1::<f32>()?,
            expected.flatten_all()?.to_vec1::<f32>()?
        );
        Ok(())
    }

    #[test]
    #[ignore = "requires a Metal device; no models"]
    fn conv_transpose_short_input_metal() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let x = Tensor::ones((1, 2, 1), DType::F32, &Device::Cpu)?;
        let w = Tensor::ones((2, 3, 4), DType::F32, &Device::Cpu)?;
        // One input frame and two unit input channels: full convolution
        // has four samples of value 2; cropping one on each side leaves two.
        // Candle's generic output-length formula itself underflows here.
        let reference = Tensor::full(2f32, (1, 3, 2), &Device::Cpu)?;
        let x = x.to_device(&dev)?;
        let w = w.to_device(&dev)?;
        let actual = conv_transpose1d_fused(&x, &w, None, 2, 1)?;
        assert_eq!(actual.dims(), reference.dims());
        assert_eq!(
            actual.flatten_all()?.to_vec1::<f32>()?,
            reference.flatten_all()?.to_vec1::<f32>()?
        );
        assert!(conv_transpose1d_fused(&x, &w, None, 0, 1).is_err());
        assert!(conv_transpose1d_fused(&x, &w, None, 2, 10).is_err());
        Ok(())
    }

    #[test]
    #[ignore = "requires a Metal device; no models"]
    fn rejects_mixed_dtype_rope_tables() -> Result<()> {
        let dev = Device::new_metal(0)?;
        let qkv = Tensor::zeros((1, 2, 1536), DType::F32, &dev)?;
        let table = Tensor::ones((2, 32), DType::F16, &dev)?;
        assert!(qkv_split_rope(&qkv, &table, &table, 8, 64).is_err());
        Ok(())
    }
}
