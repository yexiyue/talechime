//! Metal probe: where does the OmniVoice stage0 forward time go?
//!
//! Three questions, measured with the real short-sentence shapes
//! (batch 2 = CFG cond+uncond rows, sequence 157, hidden 1024, 16 q heads,
//! 8 kv heads, head_dim 128, ffn 3072, audio vocab 8 x 1025, F16):
//!
//! 1. Is a single GEMM call dominated by a fixed per-call cost?
//!    The cost is reported both pipelined (many calls, one barrier at the end)
//!    and with the host submission time separated from the GPU time.
//! 2. Does merging the q/k/v and the gate/up projections into one GEMM each
//!    reduce the time of one 28-layer step?
//! 3. How much does the padded/constant part of the sequence cost?  The same
//!    GEMM set is measured at the current M=314 and at the M values a
//!    target-only forward would use.
//!
//! The probe only uses synthetic tensors; it never loads a model.

use std::time::Instant;

use candle_core::{DType, Device, Module, Tensor};
use candle_nn::RmsNorm;

const HIDDEN: usize = 1024;
const Q_DIM: usize = 2048;
const KV_DIM: usize = 1024;
const FFN: usize = 3072;
const HEAD_DIM: usize = 128;
const LAYERS: usize = 28;
const AUDIO_VOCAB: usize = 1025;
const CODEBOOKS: usize = 8;

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
        "{label:42} submit {:9.1} us/op   gpu-total {:9.1} us/op",
        submitted / iters as f64 * 1e6,
        total / iters as f64 * 1e6,
    );
}

/// Per-call cost of one GEMM, pipelined over `iters` calls.
fn bench_gemm(device: &Device, label: &str, m: usize, n: usize, k: usize) {
    let lhs = Tensor::rand(0f32, 1f32, (m, k), device)
        .expect("lhs")
        .to_dtype(DType::F16)
        .expect("lhs f16");
    let rhs = Tensor::rand(0f32, 1f32, (k, n), device)
        .expect("rhs")
        .to_dtype(DType::F16)
        .expect("rhs f16");
    submit_per_op(device, label, 50, || {
        let _ = lhs.matmul(&rhs).expect("matmul");
    });
}

struct LayerWeights {
    q: Vec<Tensor>,
    k: Vec<Tensor>,
    v: Vec<Tensor>,
    o: Vec<Tensor>,
    gate: Vec<Tensor>,
    up: Vec<Tensor>,
    down: Vec<Tensor>,
    qkv: Vec<Tensor>,
    gate_up: Vec<Tensor>,
}

fn build_layer_weights(device: &Device) -> LayerWeights {
    let mut weights = LayerWeights {
        q: Vec::new(),
        k: Vec::new(),
        v: Vec::new(),
        o: Vec::new(),
        gate: Vec::new(),
        up: Vec::new(),
        down: Vec::new(),
        qkv: Vec::new(),
        gate_up: Vec::new(),
    };
    let f16 = |shape: (usize, usize)| {
        Tensor::rand(0f32, 1f32, shape, device)
            .expect("weight")
            .to_dtype(DType::F16)
            .expect("weight f16")
    };
    for _ in 0..LAYERS {
        weights.q.push(f16((HIDDEN, Q_DIM)));
        weights.k.push(f16((HIDDEN, KV_DIM)));
        weights.v.push(f16((HIDDEN, KV_DIM)));
        weights.o.push(f16((Q_DIM, HIDDEN)));
        weights.gate.push(f16((HIDDEN, FFN)));
        weights.up.push(f16((HIDDEN, FFN)));
        weights.down.push(f16((FFN, HIDDEN)));
        weights.qkv.push(
            Tensor::cat(
                &[
                    &weights.q[weights.q.len() - 1],
                    &weights.k[weights.k.len() - 1],
                    &weights.v[weights.v.len() - 1],
                ],
                candle_core::D::Minus1,
            )
            .expect("cat qkv"),
        );
        weights.gate_up.push(
            Tensor::cat(
                &[
                    &weights.gate[weights.gate.len() - 1],
                    &weights.up[weights.up.len() - 1],
                ],
                candle_core::D::Minus1,
            )
            .expect("cat gate_up"),
        );
    }
    weights
}

/// One 28-layer step worth of projection GEMMs.
///
/// `merged` concatenates the q/k/v weights and the gate/up weights so that
/// each projection family becomes a single GEMM.
fn bench_layer_step(device: &Device, label: &str, m: usize, merged: bool) {
    let weights = build_layer_weights(device);
    let mut hidden = Tensor::rand(0f32, 1f32, (m, HIDDEN), device)
        .expect("hidden")
        .to_dtype(DType::F16)
        .expect("hidden f16");

    let iters = 8;
    for _ in 0..3 {
        hidden = run_layer_step(&hidden, &weights, merged);
    }
    device.synchronize().expect("warmup sync");
    let began = Instant::now();
    for _ in 0..iters {
        hidden = run_layer_step(&hidden, &weights, merged);
    }
    let submitted = began.elapsed().as_secs_f64();
    device.synchronize().expect("sync");
    let total = began.elapsed().as_secs_f64();
    println!(
        "{label:42} submit {:9.3} ms/step  gpu-total {:9.3} ms/step",
        submitted / iters as f64 * 1e3,
        total / iters as f64 * 1e3,
    );
}

fn run_layer_step(hidden: &Tensor, weights: &LayerWeights, merged: bool) -> Tensor {
    let mut current = hidden.clone();
    for index in 0..LAYERS {
        // The attention output is projected by o_proj; the attention itself is
        // not part of this probe, so a fixed shape stands in for it.
        let context = if merged {
            let qkv = current.matmul(&weights.qkv[index]).expect("qkv");
            let _ = qkv
                .narrow(candle_core::D::Minus1, 0, KV_DIM)
                .expect("k part");
            let _ = qkv
                .narrow(candle_core::D::Minus1, KV_DIM, KV_DIM)
                .expect("v part");
            qkv.narrow(candle_core::D::Minus1, 0, Q_DIM)
                .expect("q part")
                .contiguous()
                .expect("q contig")
        } else {
            let _ = current.matmul(&weights.k[index]).expect("k");
            let _ = current.matmul(&weights.v[index]).expect("v");
            current.matmul(&weights.q[index]).expect("q")
        };
        let attended = context.matmul(&weights.o[index]).expect("o");

        let gate_up = if merged {
            current.matmul(&weights.gate_up[index]).expect("gate_up")
        } else {
            let gate = current.matmul(&weights.gate[index]).expect("gate");
            let up = current.matmul(&weights.up[index]).expect("up");
            Tensor::cat(&[&gate, &up], candle_core::D::Minus1).expect("cat gate_up")
        };
        let activated = gate_up
            .narrow(candle_core::D::Minus1, 0, FFN)
            .expect("gate part")
            .silu()
            .expect("silu")
            .mul(
                &gate_up
                    .narrow(candle_core::D::Minus1, FFN, FFN)
                    .expect("up part"),
            )
            .expect("mul");
        let down = activated.matmul(&weights.down[index]).expect("down");
        current = (&attended + &down).expect("residual");
    }
    current
}

/// Non-GEMM work that one decoder layer performs.
fn bench_elementwise(device: &Device, label: &str, m: usize) {
    let hidden = Tensor::rand(0f32, 1f32, (m, HIDDEN), device)
        .expect("hidden")
        .to_dtype(DType::F16)
        .expect("hidden f16");
    let norm_weight = Tensor::rand(0f32, 1f32, (HIDDEN,), device)
        .expect("norm weight")
        .to_dtype(DType::F16)
        .expect("norm weight f16");
    let norm = RmsNorm::new(norm_weight, 1e-6);

    submit_per_op(device, &format!("{label} rms_norm"), 50, || {
        let _ = norm.forward(&hidden).expect("rms norm");
    });

    let heads = 16usize;
    let seq = m / 2;
    let q = Tensor::rand(0f32, 1f32, (2, heads, seq, HEAD_DIM), device)
        .expect("q")
        .to_dtype(DType::F16)
        .expect("q f16");
    let cos = Tensor::rand(0f32, 1f32, (seq, HEAD_DIM / 2), device)
        .expect("cos")
        .to_dtype(DType::F16)
        .expect("cos f16");
    let sin = Tensor::rand(0f32, 1f32, (seq, HEAD_DIM / 2), device)
        .expect("sin")
        .to_dtype(DType::F16)
        .expect("sin f16");
    submit_per_op(device, &format!("{label} rope"), 50, || {
        let _ = candle_nn::rotary_emb::rope(&q, &cos, &sin).expect("rope");
    });

    let mlp = Tensor::rand(0f32, 1f32, (m, FFN), device)
        .expect("mlp")
        .to_dtype(DType::F16)
        .expect("mlp f16");
    submit_per_op(device, &format!("{label} silu"), 50, || {
        let _ = mlp.silu().expect("silu");
    });

    let context = Tensor::rand(0f32, 1f32, (2, heads, seq, HEAD_DIM), device)
        .expect("context")
        .to_dtype(DType::F16)
        .expect("context f16");
    submit_per_op(
        device,
        &format!("{label} context transpose+contiguous"),
        50,
        || {
            let _ = context
                .transpose(1, 2)
                .expect("transpose")
                .contiguous()
                .expect("contiguous");
        },
    );

    submit_per_op(device, &format!("{label} residual add"), 50, || {
        let _ = (&hidden + &hidden).expect("add");
    });
}

/// The audio head projection: current width and the width a target-only
/// forward would need.
fn bench_audio_head(device: &Device, label: &str, m: usize) {
    let lhs = Tensor::rand(0f32, 1f32, (m, HIDDEN), device)
        .expect("lhs")
        .to_dtype(DType::F16)
        .expect("lhs f16");
    let rhs = Tensor::rand(0f32, 1f32, (HIDDEN, CODEBOOKS * AUDIO_VOCAB), device)
        .expect("rhs")
        .to_dtype(DType::F16)
        .expect("rhs f16");
    submit_per_op(device, label, 30, || {
        let _ = lhs.matmul(&rhs).expect("audio head");
    });
}

/// Compare two tensors bit for bit (F16 -> F32 widening is exact, so the F32
/// bit pattern is a faithful proxy for the F16 one).
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
        "{label:42} differing {:8} / {:8} elements   max_abs {max_abs:.3e}",
        differing,
        lhs.len()
    );
}

/// Does merging the projections change the arithmetic at all?
fn bench_merge_equivalence(device: &Device, m: usize) {
    let x = Tensor::rand(0f32, 1f32, (m, HIDDEN), device)
        .expect("x")
        .to_dtype(DType::F16)
        .expect("x f16");
    let w_q = Tensor::rand(0f32, 1f32, (HIDDEN, Q_DIM), device)
        .expect("wq")
        .to_dtype(DType::F16)
        .expect("wq f16");
    let w_k = Tensor::rand(0f32, 1f32, (HIDDEN, KV_DIM), device)
        .expect("wk")
        .to_dtype(DType::F16)
        .expect("wk f16");
    let w_v = Tensor::rand(0f32, 1f32, (HIDDEN, KV_DIM), device)
        .expect("wv")
        .to_dtype(DType::F16)
        .expect("wv f16");
    let w_gate = Tensor::rand(0f32, 1f32, (HIDDEN, FFN), device)
        .expect("wgate")
        .to_dtype(DType::F16)
        .expect("wgate f16");
    let w_up = Tensor::rand(0f32, 1f32, (HIDDEN, FFN), device)
        .expect("wup")
        .to_dtype(DType::F16)
        .expect("wup f16");

    let q = x.matmul(&w_q).expect("q");
    let k = x.matmul(&w_k).expect("k");
    let v = x.matmul(&w_v).expect("v");
    let gate = x.matmul(&w_gate).expect("gate");
    let up = x.matmul(&w_up).expect("up");

    let w_qkv = Tensor::cat(&[&w_q, &w_k, &w_v], candle_core::D::Minus1).expect("cat qkv");
    let qkv = x.matmul(&w_qkv).expect("qkv");
    compare_bits(
        "merged qkv: q slice",
        &q,
        &qkv.narrow(candle_core::D::Minus1, 0, Q_DIM)
            .expect("q slice"),
    );
    compare_bits(
        "merged qkv: k slice",
        &k,
        &qkv.narrow(candle_core::D::Minus1, Q_DIM, KV_DIM)
            .expect("k slice"),
    );
    compare_bits(
        "merged qkv: v slice",
        &v,
        &qkv.narrow(candle_core::D::Minus1, Q_DIM + KV_DIM, KV_DIM)
            .expect("v slice"),
    );

    let w_gate_up = Tensor::cat(&[&w_gate, &w_up], candle_core::D::Minus1).expect("cat gate_up");
    let gate_up = x.matmul(&w_gate_up).expect("gate_up");
    compare_bits(
        "merged gate_up: gate slice",
        &gate,
        &gate_up
            .narrow(candle_core::D::Minus1, 0, FFN)
            .expect("gate slice"),
    );
    compare_bits(
        "merged gate_up: up slice",
        &up,
        &gate_up
            .narrow(candle_core::D::Minus1, FFN, FFN)
            .expect("up slice"),
    );
}

fn main() {
    let device = Device::new_metal(0).expect("metal device");
    println!("device: {device:?}");

    println!("\n== single GEMM: pipelined cost per call ==");
    for (m, n, k) in [
        (314usize, 1024usize, 1024usize),
        (314, 2048, 1024),
        (314, 4096, 1024),
        (314, 3072, 1024),
        (314, 6144, 1024),
        (314, 1024, 2048),
        (314, 1024, 3072),
        (314, 8200, 1024),
        (157, 1024, 1024),
        (130, 1024, 1024),
    ] {
        bench_gemm(&device, &format!("gemm M={m} N={n} K={k}"), m, n, k);
    }

    println!("\n== one 28-layer step of GEMMs ==");
    bench_layer_step(&device, "separate projections M=314", 314, false);
    bench_layer_step(&device, "merged projections  M=314", 314, true);

    println!("\n== the same step at other sequence widths ==");
    for m in [314usize, 287, 260, 157, 130] {
        bench_layer_step(&device, &format!("separate projections M={m}"), m, false);
    }

    println!("\n== non-GEMM layer work (per call) ==");
    bench_elementwise(&device, "M=314", 314);

    println!("\n== audio head ==");
    for m in [314usize, 260, 130] {
        bench_audio_head(&device, &format!("audio head M={m}"), m);
    }

    println!("\n== merged vs separate projections: bit comparison ==");
    bench_merge_equivalence(&device, 314);
}
