//! Metal probe: can the stage0 attention path avoid its layout copies?
//!
//! The current path copies the q and k projections into a head-major layout
//! before the RMS norm and the rotary embedding, because the fused rope
//! kernel only accepts a contiguous `[B, H, S, D]` tensor.  `rope_thd` accepts
//! a contiguous `[B, S, H, D]` tensor instead, which is the layout the
//! projection already produces, so the copy may become unnecessary.
//!
//! The probe checks:
//!   1. rms_norm on `[B, S, H, D]` vs on the flattened `[B*H, S, D]`
//!   2. `rope_thd` vs `rope` on the same values
//!   3. the whole q/k path: current vs copy-free
//!   4. sdpa with the strided (transposed) q/k/v, with and without a mask
//!
//! Nothing here loads a model; all tensors are synthetic.

use std::time::Instant;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::RmsNorm;

const BATCH: usize = 2;
const SEQ: usize = 157;
const HEADS: usize = 16;
const KV_HEADS: usize = 8;
const HEAD_DIM: usize = 128;

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

    let proj_q = Tensor::rand(0f32, 1f32, (BATCH, SEQ, HEADS * HEAD_DIM), &device)
        .expect("proj q")
        .to_dtype(DType::F16)
        .expect("proj q f16");
    let proj_k = Tensor::rand(0f32, 1f32, (BATCH, SEQ, KV_HEADS * HEAD_DIM), &device)
        .expect("proj k")
        .to_dtype(DType::F16)
        .expect("proj k f16");
    let proj_v = Tensor::rand(0f32, 1f32, (BATCH, SEQ, KV_HEADS * HEAD_DIM), &device)
        .expect("proj v")
        .to_dtype(DType::F16)
        .expect("proj v f16");

    let norm_weight = Tensor::rand(0f32, 1f32, (HEAD_DIM,), &device)
        .expect("norm weight")
        .to_dtype(DType::F16)
        .expect("norm weight f16");
    let norm = RmsNorm::new(norm_weight, 1e-6);
    let cos = Tensor::rand(0f32, 1f32, (SEQ, HEAD_DIM / 2), &device)
        .expect("cos")
        .to_dtype(DType::F16)
        .expect("cos f16");
    let sin = Tensor::rand(0f32, 1f32, (SEQ, HEAD_DIM / 2), &device)
        .expect("sin")
        .to_dtype(DType::F16)
        .expect("sin f16");

    println!("\n== rms_norm: flattened vs sequence-first ==");
    let q_head_major = proj_q
        .reshape((BATCH, SEQ, HEADS, HEAD_DIM))
        .expect("reshape")
        .transpose(1, 2)
        .expect("transpose");
    let flat = q_head_major.flatten(0, 2).expect("flatten");
    let norm_flat = norm.forward(&flat).expect("norm flat");
    let norm_seq_first = norm
        .forward(
            &proj_q
                .reshape((BATCH, SEQ, HEADS, HEAD_DIM))
                .expect("reshape"),
        )
        .expect("norm seq first");
    compare_bits(
        "rms_norm flat vs sequence-first",
        &norm_flat,
        &norm_seq_first
            .reshape((BATCH * HEADS, SEQ, HEAD_DIM))
            .expect("reshape"),
    );

    println!("\n== rope vs rope_thd ==");
    let q_contiguous_head_major = q_head_major.contiguous().expect("contiguous");
    let rope_head_major =
        candle_nn::rotary_emb::rope(&q_contiguous_head_major, &cos, &sin).expect("rope");
    let rope_seq_first =
        candle_nn::rotary_emb::rope_thd(&norm_seq_first, &cos, &sin).expect("rope thd");
    compare_bits(
        "rope(head-major) vs rope_thd(seq-first)",
        &rope_head_major,
        &rope_seq_first.transpose(1, 2).expect("transpose"),
    );

    println!("\n== the whole q/k path ==");
    // Current path: copy into head-major, normalise, rotate.
    let current_q = {
        let flat = q_head_major.flatten(0, 2).expect("flatten");
        let normed = norm.forward(&flat).expect("norm");
        let normed = normed
            .reshape((BATCH, HEADS, SEQ, HEAD_DIM))
            .expect("reshape");
        candle_nn::rotary_emb::rope(&normed, &cos, &sin).expect("rope")
    };
    let current_k = {
        let head_major = proj_k
            .reshape((BATCH, SEQ, KV_HEADS, HEAD_DIM))
            .expect("reshape")
            .transpose(1, 2)
            .expect("transpose");
        let flat = head_major.flatten(0, 2).expect("flatten");
        let normed = norm.forward(&flat).expect("norm");
        let normed = normed
            .reshape((BATCH, KV_HEADS, SEQ, HEAD_DIM))
            .expect("reshape");
        candle_nn::rotary_emb::rope(&normed, &cos, &sin).expect("rope")
    };
    // Copy-free path: normalise and rotate in the sequence-first layout and
    // hand the transposed view to the attention.
    let copy_free_q = {
        let normed = norm
            .forward(
                &proj_q
                    .reshape((BATCH, SEQ, HEADS, HEAD_DIM))
                    .expect("reshape"),
            )
            .expect("norm");
        candle_nn::rotary_emb::rope_thd(&normed, &cos, &sin)
            .expect("rope thd")
            .transpose(1, 2)
            .expect("transpose")
    };
    let copy_free_k = {
        let normed = norm
            .forward(
                &proj_k
                    .reshape((BATCH, SEQ, KV_HEADS, HEAD_DIM))
                    .expect("reshape"),
            )
            .expect("norm");
        candle_nn::rotary_emb::rope_thd(&normed, &cos, &sin)
            .expect("rope thd")
            .transpose(1, 2)
            .expect("transpose")
    };
    compare_bits("q: current vs copy-free", &current_q, &copy_free_q);
    compare_bits("k: current vs copy-free", &current_k, &copy_free_k);

    println!("\n== cost of the two paths ==");
    submit_per_op(&device, "current q path", 50, || {
        let flat = q_head_major.flatten(0, 2).expect("flatten");
        let normed = norm.forward(&flat).expect("norm");
        let normed = normed
            .reshape((BATCH, HEADS, SEQ, HEAD_DIM))
            .expect("reshape");
        let _ = candle_nn::rotary_emb::rope(&normed, &cos, &sin).expect("rope");
    });
    submit_per_op(&device, "copy-free q path", 50, || {
        let normed = norm
            .forward(
                &proj_q
                    .reshape((BATCH, SEQ, HEADS, HEAD_DIM))
                    .expect("reshape"),
            )
            .expect("norm");
        let _ = candle_nn::rotary_emb::rope_thd(&normed, &cos, &sin)
            .expect("rope thd")
            .transpose(1, 2)
            .expect("transpose");
    });

    println!("\n== sdpa with the strided q/k/v, with and without a mask ==");
    let v_strided = proj_v
        .reshape((BATCH, SEQ, KV_HEADS, HEAD_DIM))
        .expect("reshape")
        .transpose(1, 2)
        .expect("transpose");
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let mask = Tensor::rand(0f32, 1f32, (BATCH, 1, SEQ, SEQ), &device)
        .expect("mask")
        .to_dtype(DType::F16)
        .expect("mask f16");
    let mask_expanded = mask.expand((BATCH, HEADS, SEQ, SEQ)).expect("expand");

    for (label, use_mask) in [("no mask", false), ("with mask", true)] {
        let mask_ref = if use_mask { Some(&mask_expanded) } else { None };
        let contiguous = candle_nn::ops::sdpa(
            &current_q,
            &current_k,
            &v_strided.contiguous().expect("v contiguous"),
            mask_ref,
            false,
            scale,
            1.0,
        )
        .expect("sdpa");
        let strided = candle_nn::ops::sdpa(
            &copy_free_q,
            &copy_free_k,
            &v_strided,
            mask_ref,
            false,
            scale,
            1.0,
        )
        .expect("sdpa");
        compare_bits(
            &format!("sdpa {label}: current vs copy-free"),
            &contiguous,
            &strided,
        );
    }

    println!("\n== cost of sdpa with contiguous vs strided inputs ==");
    let v_contiguous = v_strided.contiguous().expect("v contiguous");
    submit_per_op(&device, "sdpa contiguous q/k", 50, || {
        let _ = candle_nn::ops::sdpa(
            &current_q,
            &current_k,
            &v_contiguous,
            Some(&mask_expanded),
            false,
            scale,
            1.0,
        )
        .expect("sdpa");
    });
    submit_per_op(&device, "sdpa strided q/k", 50, || {
        let _ = candle_nn::ops::sdpa(
            &copy_free_q,
            &copy_free_k,
            &v_strided,
            Some(&mask_expanded),
            false,
            scale,
            1.0,
        )
        .expect("sdpa");
    });
}
