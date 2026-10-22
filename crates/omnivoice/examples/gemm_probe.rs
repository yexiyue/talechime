//! Metal micro-benchmark for the OmniVoice stage0 shapes.
//!
//! Mirrors the PyTorch MPS GEMM probe so the two backends can be compared on
//! identical matrix shapes and dtypes.

use std::time::Instant;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::RmsNorm;

fn bench_gemm(device: &Device, label: &str, m: usize, n: usize, k: usize) {
    let a = Tensor::rand(0f32, 1f32, (m, k), device)
        .expect("lhs")
        .to_dtype(DType::F16)
        .expect("lhs f16");
    let b = Tensor::rand(0f32, 1f32, (k, n), device)
        .expect("rhs")
        .to_dtype(DType::F16)
        .expect("rhs f16");
    for _ in 0..10 {
        let _ = a.matmul(&b).expect("warmup matmul");
    }
    device.synchronize().expect("warmup sync");
    let iters = 50;
    let began = Instant::now();
    for _ in 0..iters {
        let _ = a.matmul(&b).expect("matmul");
    }
    device.synchronize().expect("sync");
    let per = began.elapsed().as_secs_f64() / iters as f64;
    let tflops = 2.0 * m as f64 * n as f64 * k as f64 / per / 1e12;
    println!(
        "{label:14} M={m:5} N={n:5} K={k:5} {:8.1} us {tflops:6.2} TFLOP/s",
        per * 1e6
    );
}

fn bench_attention(device: &Device, label: &str, batch: usize, heads: usize, seq: usize) {
    let dim = 128usize;
    let q = Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), device)
        .expect("q")
        .to_dtype(DType::F16)
        .expect("q f16");
    let k = Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), device)
        .expect("k")
        .to_dtype(DType::F16)
        .expect("k f16");
    let v = Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), device)
        .expect("v")
        .to_dtype(DType::F16)
        .expect("v f16");
    let scale = 1.0f32 / (dim as f32).sqrt();
    for _ in 0..10 {
        let _ = manual_attention(&q, &k, &v, scale).expect("warmup attention");
    }
    device.synchronize().expect("warmup sync");
    let iters = 50;
    let began = Instant::now();
    for _ in 0..iters {
        let _ = manual_attention(&q, &k, &v, scale).expect("attention");
    }
    device.synchronize().expect("sync");
    let per = began.elapsed().as_secs_f64() / iters as f64;
    println!(
        "{label:14} B={batch} H={heads} S={seq:5} {:8.1} us",
        per * 1e6
    );
}

fn bench_attention_masked(device: &Device, label: &str, batch: usize, heads: usize, seq: usize) {
    let dim = 128usize;
    let kv_heads = 8usize;
    let q = Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), device)
        .expect("q")
        .to_dtype(DType::F16)
        .expect("q f16");
    let k = Tensor::rand(0f32, 1f32, (batch, kv_heads, seq, dim), device)
        .expect("k")
        .to_dtype(DType::F16)
        .expect("k f16");
    let v = Tensor::rand(0f32, 1f32, (batch, kv_heads, seq, dim), device)
        .expect("v")
        .to_dtype(DType::F16)
        .expect("v f16");
    let mask = Tensor::rand(0f32, 1f32, (batch, 1, seq, seq), device)
        .expect("mask")
        .to_dtype(DType::F16)
        .expect("mask f16");
    let scale = 1.0f32 / (dim as f32).sqrt();
    let reps = heads / kv_heads;
    for _ in 0..10 {
        let _ = manual_attention_masked(&q, &k, &v, &mask, scale, reps).expect("warmup");
    }
    device.synchronize().expect("warmup sync");
    let iters = 50;
    let began = Instant::now();
    for _ in 0..iters {
        let _ = manual_attention_masked(&q, &k, &v, &mask, scale, reps).expect("attention");
    }
    device.synchronize().expect("sync");
    let per = began.elapsed().as_secs_f64() / iters as f64;
    println!(
        "{label:14} B={batch} H={heads} S={seq:5} {:8.1} us",
        per * 1e6
    );
}

fn manual_attention_masked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: &Tensor,
    scale: f32,
    reps: usize,
) -> candle_core::Result<Tensor> {
    let kk = candle_transformers::utils::repeat_kv(k.clone(), reps)?;
    let vv = candle_transformers::utils::repeat_kv(v.clone(), reps)?;
    let k_t = kk.transpose(2, 3)?.contiguous()?;
    let scores = q.matmul(&k_t)?.affine(f64::from(scale), 0.0)?;
    let scores = scores.broadcast_add(mask)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    probs.matmul(&vv)
}

fn bench_sdpa(device: &Device, label: &str, batch: usize, heads: usize, seq: usize) {
    let dim = 128usize;
    let kv_heads = 8usize;
    let q = Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), device)
        .expect("q")
        .to_dtype(DType::F16)
        .expect("q f16");
    let k = Tensor::rand(0f32, 1f32, (batch, kv_heads, seq, dim), device)
        .expect("k")
        .to_dtype(DType::F16)
        .expect("k f16");
    let v = Tensor::rand(0f32, 1f32, (batch, kv_heads, seq, dim), device)
        .expect("v")
        .to_dtype(DType::F16)
        .expect("v f16");
    let mask = Tensor::rand(0f32, 1f32, (batch, 1, seq, seq), device)
        .expect("mask")
        .to_dtype(DType::F16)
        .expect("mask f16");
    // Expand (not materialize) the head dimension: the fused kernel reads the
    // mask through its strides, so a zero head stride keeps the [B,1,S,S] mask.
    let mask = mask
        .expand((batch, heads, seq, seq))
        .expect("mask expand")
        .contiguous()
        .expect("mask contig");
    let scale = 1.0f32 / (dim as f32).sqrt();
    let reps = heads / kv_heads;
    for _ in 0..10 {
        let kk = candle_transformers::utils::repeat_kv(k.clone(), reps).expect("repeat_kv");
        let vv = candle_transformers::utils::repeat_kv(v.clone(), reps).expect("repeat_kv");
        let _ = candle_nn::ops::sdpa(&q, &kk, &vv, Some(&mask), false, scale, 1.0)
            .expect("sdpa warmup");
    }
    device.synchronize().expect("warmup sync");
    let iters = 50;
    let began = Instant::now();
    for _ in 0..iters {
        let kk = candle_transformers::utils::repeat_kv(k.clone(), reps).expect("repeat_kv");
        let vv = candle_transformers::utils::repeat_kv(v.clone(), reps).expect("repeat_kv");
        let _ = candle_nn::ops::sdpa(&q, &kk, &vv, Some(&mask), false, scale, 1.0).expect("sdpa");
    }
    device.synchronize().expect("sync");
    let per = began.elapsed().as_secs_f64() / iters as f64;
    println!(
        "{label:14} B={batch} H={heads} S={seq:5} {:8.1} us",
        per * 1e6
    );
}

fn timed<F: FnMut()>(device: &Device, label: &str, iters: usize, mut f: F) {
    for _ in 0..5 {
        f();
    }
    device.synchronize().expect("warmup sync");
    let began = Instant::now();
    for _ in 0..iters {
        f();
    }
    device.synchronize().expect("sync");
    let per = began.elapsed().as_secs_f64() / iters as f64;
    println!("{label:24} {:8.1} us", per * 1e6);
}

fn bench_layer_parts(device: &Device, seq: usize) {
    let batch = 2usize;
    let hidden = 1024usize;
    let heads = 16usize;
    let kv_heads = 8usize;
    let dim = 128usize;
    let tokens = batch * seq;

    let xs = Tensor::rand(0f32, 1f32, (batch, seq, hidden), device)
        .expect("xs")
        .to_dtype(DType::F16)
        .expect("xs f16");
    let flat = xs
        .reshape((tokens, hidden))
        .expect("reshape")
        .contiguous()
        .expect("contig");
    let weight = Tensor::rand(0f32, 1f32, (hidden, hidden), device)
        .expect("weight")
        .to_dtype(DType::F16)
        .expect("weight f16");
    let bias = Tensor::rand(0f32, 1f32, (hidden,), device)
        .expect("bias")
        .to_dtype(DType::F16)
        .expect("bias f16");
    let norm = RmsNorm::new(
        Tensor::rand(0f32, 1f32, (hidden,), device)
            .expect("norm weight")
            .to_dtype(DType::F16)
            .expect("norm weight f16"),
        1e-6,
    );

    let q = Tensor::rand(0f32, 1f32, (batch, heads, seq, dim), device)
        .expect("q")
        .to_dtype(DType::F16)
        .expect("q f16");
    let k = Tensor::rand(0f32, 1f32, (batch, kv_heads, seq, dim), device)
        .expect("k")
        .to_dtype(DType::F16)
        .expect("k f16");
    let v = Tensor::rand(0f32, 1f32, (batch, kv_heads, seq, dim), device)
        .expect("v")
        .to_dtype(DType::F16)
        .expect("v f16");
    let mask = Tensor::rand(0f32, 1f32, (batch, 1, seq, seq), device)
        .expect("mask")
        .to_dtype(DType::F16)
        .expect("mask f16");
    let sin = Tensor::rand(0f32, 1f32, (seq, dim / 2), device)
        .expect("sin")
        .to_dtype(DType::F16)
        .expect("sin f16");
    let cos = Tensor::rand(0f32, 1f32, (seq, dim / 2), device)
        .expect("cos")
        .to_dtype(DType::F16)
        .expect("cos f16");

    let reps = heads / kv_heads;
    println!("--- per-op, seq={seq} tokens={tokens} ---");
    timed(device, "linear(1024x1024)", 100, || {
        let _ = flat.matmul(&weight).expect("matmul");
    });
    timed(device, "linear+bias", 100, || {
        let out = flat.matmul(&weight).expect("matmul");
        let _ = out.broadcast_add(&bias).expect("bias");
    });
    timed(device, "rms_norm f16", 100, || {
        let _ = norm.forward_diff(&flat).expect("rms");
    });
    timed(device, "rms_norm fused f16", 100, || {
        let _ = norm.forward(&flat).expect("rms");
    });
    timed(device, "silu", 100, || {
        let _ = flat.broadcast_mul(&flat).expect("silu");
    });
    timed(device, "rope", 100, || {
        let _ = candle_nn::rotary_emb::rope(&q, &cos, &sin).expect("rope");
    });
    timed(device, "repeat_kv", 100, || {
        let _ = candle_transformers::utils::repeat_kv(k.clone(), reps).expect("repeat_kv");
    });
    timed(device, "attn no-mask", 100, || {
        let kk = candle_transformers::utils::repeat_kv(k.clone(), reps).expect("repeat_kv");
        let vv = candle_transformers::utils::repeat_kv(v.clone(), reps).expect("repeat_kv");
        let k_t = kk
            .transpose(2, 3)
            .expect("kt")
            .contiguous()
            .expect("contig");
        let scores = q.matmul(&k_t).expect("scores");
        let probs = candle_nn::ops::softmax_last_dim(&scores).expect("softmax");
        let _ = probs.matmul(&vv).expect("av");
    });
    timed(device, "attn with-mask", 100, || {
        let kk = candle_transformers::utils::repeat_kv(k.clone(), reps).expect("repeat_kv");
        let vv = candle_transformers::utils::repeat_kv(v.clone(), reps).expect("repeat_kv");
        let k_t = kk
            .transpose(2, 3)
            .expect("kt")
            .contiguous()
            .expect("contig");
        let scores = q.matmul(&k_t).expect("scores");
        let scores = scores.broadcast_add(&mask).expect("mask");
        let probs = candle_nn::ops::softmax_last_dim(&scores).expect("softmax");
        let _ = probs.matmul(&vv).expect("av");
    });
    timed(device, "transpose+contig", 100, || {
        let _ = k.transpose(2, 3).expect("kt").contiguous().expect("contig");
    });
    let scores = Tensor::rand(0f32, 1f32, (batch, heads, seq, seq), device)
        .expect("scores")
        .to_dtype(DType::F16)
        .expect("scores f16");
    timed(device, "broadcast_add [2,1,S,S]", 100, || {
        let _ = scores.broadcast_add(&mask).expect("mask add");
    });
    let mask_full = Tensor::rand(0f32, 1f32, (batch, heads, seq, seq), device)
        .expect("mask full")
        .to_dtype(DType::F16)
        .expect("mask full f16");
    timed(device, "broadcast_add full", 100, || {
        let _ = scores.broadcast_add(&mask_full).expect("mask add");
    });
    timed(device, "silu 3072", 100, || {
        let m = Tensor::rand(0f32, 1f32, (tokens, 3072), device)
            .expect("m")
            .to_dtype(DType::F16)
            .expect("m f16");
        let _ = candle_nn::Activation::Silu.forward(&m).expect("silu");
    });
}

fn manual_attention(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> candle_core::Result<Tensor> {
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let scores = q.matmul(&k_t)?.affine(f64::from(scale), 0.0)?;
    let probs = candle_nn::ops::softmax_last_dim(&scores)?;
    probs.matmul(v)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = Device::new_metal(0)?;
    println!("device: {device:?} dtype: F16");
    for (label, m) in [("short", 300usize), ("long", 2800usize)] {
        bench_gemm(&device, &format!("{label} qkv_o"), m, 1024, 1024);
        bench_gemm(&device, &format!("{label} gate_up"), m, 3072, 1024);
        bench_gemm(&device, &format!("{label} down"), m, 1024, 3072);
        bench_gemm(&device, &format!("{label} audio_head"), m, 8200, 1024);
        bench_attention(&device, &format!("{label} attn"), 2, 16, m / 2);
        bench_sdpa(&device, &format!("{label} sdpa"), 2, 16, m / 2);
    }
    bench_attention(&device, "corpus attn", 2, 16, 287);
    bench_attention_masked(&device, "corpus attn+mask", 2, 16, 287);
    bench_sdpa(&device, "corpus sdpa", 2, 16, 287);
    bench_layer_parts(&device, 150);
    Ok(())
}
