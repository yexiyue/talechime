//! Private, ignored real-model capture/replay tests; no production switches.
use std::{cell::Cell, fs};

use super::*;
use crate::cpu_attribution as io;
use candle_core::Device;

thread_local! {
    static FORWARD: Cell<Option<usize>> = const { Cell::new(None) };
}

pub(super) fn begin_forward() -> Option<usize> {
    FORWARD.with(|counter| {
        let step = counter.get()?;
        counter.set(Some(step + 1));
        [0, 15, 31].contains(&step).then_some(step)
    })
}

pub(super) fn capture_layer(
    step: Option<usize>,
    layer: usize,
    input: &Tensor,
    output: &Tensor,
    mask: Option<&Tensor>,
) -> Result<()> {
    if let Some(step) = step.filter(|_| [0, 13, 27].contains(&layer)) {
        let mask =
            mask.ok_or_else(|| crate::OmniVoiceError::InvalidData("capture requires mask".into()))?;
        io::save(
            &format!("step-{step}-layer-{layer}"),
            &[("input", input), ("output", output), ("mask", mask)],
        )?;
    }
    Ok(())
}

pub(super) fn capture_head(step: Option<usize>, hidden: &Tensor) -> Result<()> {
    if let Some(step) = step {
        io::save(&format!("step-{step}-head"), &[("input", hidden)])?;
    }
    Ok(())
}

#[test]
#[ignore = "requires explicit local model, corpus and fresh output directory"]
fn capture_real_cpu_inputs() -> Result<()> {
    let text = fs::read_to_string(
        std::env::var("OMNI_CPU_TRACE_TEXT")
            .map_err(|e| crate::OmniVoiceError::InvalidRequest(e.to_string()))?,
    )?;
    let pipeline = crate::pipeline::Pipeline::from_options(
        crate::RuntimeOptions::new(io::model()?).with_device(crate::DeviceSpec::Cpu),
    )?;
    // Match talechime-core SeedPolicy::Pinned(42).resolve(0, 0), not a raw model seed.
    use std::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    42u64.hash(&mut hasher);
    0u64.hash(&mut hasher);
    0u8.hash(&mut hasher);
    let seed = hasher.finish();
    println!(
        "CPU_TRACE {}",
        serde_json::json!({"kind":"capture_seed","pinned_seed":42,"call":0,"attempt":0,"resolved_seed":seed})
    );
    pipeline.set_seed(seed)?;
    FORWARD.with(|counter| counter.set(Some(0)));
    let result = pipeline.generate(
        &crate::contracts::GenerationRequest::new_text_only(text.trim()).with_language("zh"),
    );
    let forwards = FORWARD.with(|counter| counter.replace(None));
    let audio = result?;
    assert_eq!(
        forwards,
        Some(32),
        "capture must contain one unchunked 32-step request"
    );
    for (index, audio) in audio.iter().enumerate() {
        let samples = Tensor::from_vec(audio.samples.clone(), audio.samples.len(), &Device::Cpu)?;
        io::save(&format!("capture-audio-{index}"), &[("samples", &samples)])?;
    }
    Ok(())
}

fn config() -> Result<Stage0Qwen3BackboneConfig> {
    let cfg = crate::stage0_model::Stage0Config::from_model_root(io::model()?)?.llm_config;
    Ok(Stage0Qwen3BackboneConfig {
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        intermediate_size: cfg.intermediate_size,
        num_hidden_layers: cfg.num_hidden_layers,
        num_attention_heads: cfg.num_attention_heads,
        num_key_value_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        max_position_embeddings: cfg.max_position_embeddings,
        attention_bias: cfg.attention_bias,
        rms_norm_eps: cfg.rms_norm_eps,
        rope_theta: cfg.rope_parameters.rope_theta,
        hidden_act: Activation::Silu,
    })
}

fn normalized_rotated(
    attn: &Stage0Qwen3Attention,
    q: &Tensor,
    k: &Tensor,
) -> Result<(Tensor, Tensor)> {
    let (b, s, _) = q.dims3()?;
    let q = q
        .reshape((b, s, attn.num_heads, attn.head_dim))?
        .transpose(1, 2)?;
    let k = k
        .reshape((b, s, attn.num_kv_heads, attn.head_dim))?
        .transpose(1, 2)?;
    let q = apply_rms_norm(&attn.q_norm, &q.flatten(0, 2)?)?.reshape((
        b,
        attn.num_heads,
        s,
        attn.head_dim,
    ))?;
    let k = apply_rms_norm(&attn.k_norm, &k.flatten(0, 2)?)?.reshape((
        b,
        attn.num_kv_heads,
        s,
        attn.head_dim,
    ))?;
    attn.rotary_emb.apply(&q, &k, 0)
}

#[test]
#[ignore = "requires captured local CPU inputs and immutable weights"]
fn replay_real_cpu_inputs() -> Result<()> {
    assert!(candle_core::utils::has_accelerate());
    let model = io::model()?;
    let cfg = config()?;
    // SAFETY: local pinned weights must remain immutable during this offline process.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(
            &[model.join("model.safetensors")],
            DType::F32,
            &Device::Cpu,
        )?
    };
    let rotary = Arc::new(Stage0Qwen3RotaryEmbedding::new(DType::F32, &cfg, &vb)?);
    let export = std::env::var_os("OMNI_CPU_TRACE_EXPORT").is_some();
    let reverse = std::env::var("OMNI_CPU_TRACE_REVERSE").is_ok_and(|v| v == "1");
    for layer_index in [0, 13, 27] {
        let layer = Stage0Qwen3DecoderLayer::load(
            &cfg,
            rotary.clone(),
            vb.pp(format!("llm.layers.{layer_index}")),
        )?;
        for step in [0, 15, 31] {
            let name = format!("step-{step}-layer-{layer_index}");
            let tensors = io::load(&name)?;
            let xs = &tensors["input"];
            let mask = &tensors["mask"];
            io::equal(&layer.forward(xs, Some(mask), None)?, &tensors["output"])?;
            let norm = apply_rms_norm(&layer.input_layernorm, xs)?;
            let attn = &layer.self_attn;
            let qr = norm.apply(&attn.q_proj)?;
            let kr = norm.apply(&attn.k_proj)?;
            let vr = norm.apply(&attn.v_proj)?;
            let (q, k) = normalized_rotated(attn, &qr, &kr)?;
            let (b, s, _) = xs.dims3()?;
            let v = vr
                .reshape((b, s, attn.num_kv_heads, attn.head_dim))?
                .transpose(1, 2)?;
            let context =
                attn.attention_context(q.clone(), k.clone(), v.clone(), Some(mask), None)?;
            let consumed = context
                .transpose(1, 2)?
                .contiguous()?
                .reshape((b, s, attn.hidden_size))?
                .apply(&attn.o_proj)?;
            io::equal(&consumed, &attn.forward(&norm, Some(mask), None)?)?;
            let residual = (&consumed + xs)?;
            let mlp_input = apply_rms_norm(&layer.post_attention_layernorm, &residual)?;
            let mlp_output = layer.mlp.forward(&mlp_input)?;
            let gate_up = layer.mlp.gate_up_proj.as_ref().ok_or_else(|| {
                crate::OmniVoiceError::InvalidData("expected CPU merged MLP".into())
            })?;
            let both = mlp_input.apply(gate_up)?;
            let gate = both.narrow(2, 0, 3072)?;
            let up = both.narrow(2, 3072, 3072)?;
            let down_input = (gate.silu()? * &up)?;
            io::equal(&down_input.apply(&layer.mlp.down_proj)?, &mlp_output)?;
            io::equal(&(&mlp_output + &residual)?, &tensors["output"])?;
            if export {
                io::save(
                    &format!("{name}-replay"),
                    &[
                        ("norm", &norm),
                        ("q_raw", &qr),
                        ("k_raw", &kr),
                        ("v_raw", &vr),
                        ("q", &q),
                        ("k", &k),
                        ("v", &v),
                        ("context", &context),
                        ("attention_output", &consumed),
                        ("residual", &residual),
                        ("mlp_input", &mlp_input),
                        ("mlp_output", &mlp_output),
                        ("gate", &gate),
                        ("up", &up),
                        ("down_input", &down_input),
                    ],
                )?;
            }
            let mut ops = vec![
                "layer",
                "input_norm",
                "q_proj",
                "k_proj",
                "v_proj",
                "norm_rope",
                "attention",
                "attention_consumer",
                "context_layout",
                "mlp",
                "mlp_consumer",
                "mlp_gate_up",
                "mlp_activation",
                "down_proj",
            ];
            if reverse {
                ops.reverse();
            }
            for op in ops {
                io::bench(&name, op, || {
                    Ok(match op {
                        "layer" => layer.forward(xs, Some(mask), None)?,
                        "input_norm" => apply_rms_norm(&layer.input_layernorm, xs)?,
                        "q_proj" => norm.apply(&attn.q_proj)?,
                        "k_proj" => norm.apply(&attn.k_proj)?,
                        "v_proj" => norm.apply(&attn.v_proj)?,
                        "norm_rope" => {
                            let (q, k) = normalized_rotated(attn, &qr, &kr)?;
                            let _ = std::hint::black_box(k);
                            q
                        }
                        "attention" => attn.attention_context(
                            q.clone(),
                            k.clone(),
                            v.clone(),
                            Some(mask),
                            None,
                        )?,
                        "attention_consumer" => attn.forward(&norm, Some(mask), None)?,
                        "context_layout" => context.transpose(1, 2)?.contiguous()?.reshape((
                            b,
                            s,
                            attn.hidden_size,
                        ))?,
                        "mlp" => layer.mlp.forward(&mlp_input)?,
                        "mlp_gate_up" => mlp_input.apply(gate_up)?,
                        "mlp_activation" => (gate.silu()? * &up)?,
                        "down_proj" => down_input.apply(&layer.mlp.down_proj)?,
                        "mlp_consumer" => (layer.mlp.forward(&mlp_input)? + &residual)?,
                        _ => unreachable!(),
                    })
                })?;
            }
        }
    }
    let head = candle_nn::Linear::new(vb.get((8200, 1024), "audio_heads.weight")?, None);
    for step in [0, 15, 31] {
        let name = format!("step-{step}-head");
        let t = io::load(&name)?;
        let x = &t["input"];
        let output = head.forward(x)?;
        if export {
            io::save(&format!("{name}-replay"), &[("output", &output)])?;
        }
        io::bench(&name, "output_head", || Ok(head.forward(x)?))?;
    }
    Ok(())
}
