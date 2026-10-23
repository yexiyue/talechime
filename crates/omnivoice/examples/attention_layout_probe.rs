//! Metal probe: the layout copies inside the OmniVoice stage0 attention path.
//!
//! The stage0 attention sub-path currently performs several layout copies per
//! decoder layer.  This probe measures each of them separately at the real
//! short-sentence shape (batch 2, sequence 157, 16 heads, head_dim 128, F16)
//! and compares alternative ways of producing the same values.
//!
//! Nothing here loads a model; all tensors are synthetic.

use std::time::Instant;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::RmsNorm;

const BATCH: usize = 2;
const SEQ: usize = 157;
const HEADS: usize = 16;
const HEAD_DIM: usize = 128;
const KV_HEADS: usize = 8;

/// Seconds per operation, from a host-side submission clock.
fn submit_per_op<F: FnMut()>(device: &Device, label: &str, iters: usize, mut f: F) {
    for _ in 0..10 {
        f();
    }
    device.synchronize().expect("warmup sync");
    let began = Instant::now();
    for _ in 0..iters {
        f();
    }
    let submitted = began.elapsed().as_secs_f64();
    device.synchronize().expect("sync");
    let total = began.elapsed().as_secs_f64();
    println!(
        "{label:46} submit {:8.1} us/op   gpu-total {:8.1} us/op",
        submitted / iters as f64 * 1e6,
        total / iters as f64 * 1e6,
    );
}

fn head_view(proj: &Tensor) -> Tensor {
    proj.reshape((BATCH, SEQ, HEADS, HEAD_DIM))
        .expect("reshape")
        .transpose(1, 2)
        .expect("transpose")
}

fn compare_bits(label: &str, lhs: &Tensor, rhs: &Tensor) {
    let lhs = lhs
        .to_dtype(DType::F32)
        .expect("lhs f32")
        .to_device(&Device::Cpu)
        .expect("lhs cpu")
        .flatten_all()
        .expect("lhs flat")
        .to_vec1::<f32>()
        .expect("lhs vec");
    let rhs = rhs
        .to_dtype(DType::F32)
        .expect("rhs f32")
        .to_device(&Device::Cpu)
        .expect("rhs cpu")
        .flatten_all()
        .expect("rhs flat")
        .to_vec1::<f32>()
        .expect("rhs vec");
    let mut differing = 0usize;
    let mut max_abs = 0f32;
    for (a, b) in lhs.iter().zip(rhs.iter()) {
        if a.to_bits() != b.to_bits() {
            differing += 1;
            max_abs = max_abs.max((a - b).abs());
        }
    }
    println!(
        "{label:46} differing {:8} / {:8} elements   max_abs {max_abs:.3e}",
        differing,
        lhs.len()
    );
}

fn main() {
    let device = Device::new_metal(0).expect("metal device");
    println!("device: {device:?}");

    // q-like projection output: [BATCH, SEQ, HEADS * HEAD_DIM], F16.
    let proj = Tensor::rand(0f32, 1f32, (BATCH, SEQ, HEADS * HEAD_DIM), &device)
        .expect("proj")
        .to_dtype(DType::F16)
        .expect("proj f16");
    // kv-like projection output: [BATCH, SEQ, KV_HEADS * HEAD_DIM].
    let proj_kv = Tensor::rand(0f32, 1f32, (BATCH, SEQ, KV_HEADS * HEAD_DIM), &device)
        .expect("proj kv")
        .to_dtype(DType::F16)
        .expect("proj kv f16");

    let q_view = head_view(&proj);
    let k_view = proj_kv
        .reshape((BATCH, SEQ, KV_HEADS, HEAD_DIM))
        .expect("reshape")
        .transpose(1, 2)
        .expect("transpose");
    let v_view = k_view.clone();

    println!("\n== producing a contiguous head-major tensor from the projection ==");
    submit_per_op(&device, "q view.contiguous()", 50, || {
        let _ = q_view.contiguous().expect("contiguous");
    });
    submit_per_op(&device, "q view.flatten(0,2)", 50, || {
        let _ = q_view.flatten(0, 2).expect("flatten");
    });
    submit_per_op(&device, "k view.flatten(0,2)", 50, || {
        let _ = k_view.flatten(0, 2).expect("flatten");
    });

    println!("\n== rms norm on the flattened view vs the head-major view ==");
    let norm_weight = Tensor::rand(0f32, 1f32, (HEAD_DIM,), &device)
        .expect("norm weight")
        .to_dtype(DType::F16)
        .expect("norm weight f16");
    let norm = RmsNorm::new(norm_weight, 1e-6);
    let q_flat = q_view.flatten(0, 2).expect("q flat");
    submit_per_op(&device, "rms_norm on [B*H,S,D] contiguous", 50, || {
        let _ = norm.forward(&q_flat).expect("rms norm");
    });
    submit_per_op(&device, "rms_norm on [B,H,S,D] strided view", 50, || {
        let _ = norm.forward(&q_view).expect("rms norm");
    });
    let flat_norm = norm.forward(&q_flat).expect("flat norm");
    let view_norm = norm.forward(&q_view).expect("view norm");
    compare_bits(
        "rms_norm flat vs strided view",
        &flat_norm,
        &view_norm
            .reshape((BATCH * HEADS, SEQ, HEAD_DIM))
            .expect("reshape"),
    );

    println!("\n== rope ==");
    let cos = Tensor::rand(0f32, 1f32, (SEQ, HEAD_DIM / 2), &device)
        .expect("cos")
        .to_dtype(DType::F16)
        .expect("cos f16");
    let sin = Tensor::rand(0f32, 1f32, (SEQ, HEAD_DIM / 2), &device)
        .expect("sin")
        .to_dtype(DType::F16)
        .expect("sin f16");
    let q_contiguous = q_view.contiguous().expect("q contiguous");
    submit_per_op(&device, "rope on contiguous [B,H,S,D]", 50, || {
        let _ = candle_nn::rotary_emb::rope(&q_contiguous, &cos, &sin).expect("rope");
    });
    // The fused rope kernel requires a contiguous input, so the strided view
    // cannot be used directly.
    println!("(rope on a strided view is rejected by the kernel)");

    println!("\n== sdpa with contiguous vs strided inputs ==");
    let k_contiguous = k_view.contiguous().expect("k contiguous");
    let v_contiguous = v_view.contiguous().expect("v contiguous");
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    submit_per_op(&device, "sdpa contiguous q/k/v", 50, || {
        let _ = candle_nn::ops::sdpa(
            &q_contiguous,
            &k_contiguous,
            &v_contiguous,
            None,
            false,
            scale,
            1.0,
        )
        .expect("sdpa");
    });
    submit_per_op(&device, "sdpa strided q, contiguous k/v", 50, || {
        let _ = candle_nn::ops::sdpa(
            &q_view,
            &k_contiguous,
            &v_contiguous,
            None,
            false,
            scale,
            1.0,
        )
        .expect("sdpa");
    });
    submit_per_op(&device, "sdpa strided q/k/v", 50, || {
        let _ =
            candle_nn::ops::sdpa(&q_view, &k_view, &v_view, None, false, scale, 1.0).expect("sdpa");
    });
    let sdpa_contiguous = candle_nn::ops::sdpa(
        &q_contiguous,
        &k_contiguous,
        &v_contiguous,
        None,
        false,
        scale,
        1.0,
    )
    .expect("sdpa");
    let sdpa_strided =
        candle_nn::ops::sdpa(&q_view, &k_view, &v_view, None, false, scale, 1.0).expect("sdpa");
    compare_bits(
        "sdpa contiguous vs strided",
        &sdpa_contiguous,
        &sdpa_strided,
    );

    println!("\n== moving the attention output into the o_proj layout ==");
    let context = sdpa_contiguous;
    submit_per_op(&device, "context.transpose(1,2).contiguous()", 50, || {
        let _ = context
            .transpose(1, 2)
            .expect("transpose")
            .contiguous()
            .expect("contiguous");
    });
    submit_per_op(&device, "context.permute(0,2,1,3).contiguous()", 50, || {
        let _ = context
            .permute((0, 2, 1, 3))
            .expect("permute")
            .contiguous()
            .expect("contiguous");
    });
    submit_per_op(
        &device,
        "context transpose(1,2) then reshape only",
        50,
        || {
            let _ =
                context
                    .transpose(1, 2)
                    .expect("transpose")
                    .reshape((BATCH, SEQ, HEADS * HEAD_DIM));
        },
    );
    submit_per_op(&device, "context narrow+cat over seq", 50, || {
        let mut parts = Vec::with_capacity(SEQ);
        for seq in 0..SEQ {
            parts.push(
                context
                    .narrow(2, seq, 1)
                    .expect("narrow")
                    .transpose(1, 2)
                    .expect("transpose"),
            );
        }
        let _ = Tensor::cat(&parts, candle_core::D::Minus1).expect("cat");
    });

    println!("\n== the whole attention sub-path ==");
    let o_weight = Tensor::rand(0f32, 1f32, (HEADS * HEAD_DIM, HEAD_DIM), &device)
        .expect("o weight")
        .to_dtype(DType::F16)
        .expect("o weight f16");
    let _ = o_weight;
    submit_per_op(&device, "current layout path", 30, || {
        let q = head_view(&proj);
        let k = proj_kv
            .reshape((BATCH, SEQ, KV_HEADS, HEAD_DIM))
            .expect("reshape")
            .transpose(1, 2)
            .expect("transpose");
        let v = k.clone();
        let q_flat = q.flatten(0, 2).expect("q flat");
        let k_flat = k.flatten(0, 2).expect("k flat");
        let v_flat = v.flatten(0, 2).expect("v flat");
        let q = norm.forward(&q_flat).expect("q norm");
        let k = norm.forward(&k_flat).expect("k norm");
        let q = q.reshape((BATCH, HEADS, SEQ, HEAD_DIM)).expect("q reshape");
        let k = k
            .reshape((BATCH, KV_HEADS, SEQ, HEAD_DIM))
            .expect("k reshape");
        let q = candle_nn::rotary_emb::rope(&q, &cos, &sin).expect("q rope");
        let k = candle_nn::rotary_emb::rope(&k, &cos, &sin).expect("k rope");
        let context = candle_nn::ops::sdpa(&q, &k, &v_flat, None, false, scale, 1.0).expect("sdpa");
        let _ = context
            .transpose(1, 2)
            .expect("transpose")
            .contiguous()
            .expect("contiguous")
            .reshape((BATCH, SEQ, HEADS * HEAD_DIM))
            .expect("reshape");
    });
}
