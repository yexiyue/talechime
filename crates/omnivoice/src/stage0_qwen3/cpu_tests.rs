use super::*;
use candle_core::Device;

fn mlp_config() -> Stage0Qwen3BackboneConfig {
    Stage0Qwen3BackboneConfig {
        vocab_size: 16,
        hidden_size: 1024,
        intermediate_size: 3072,
        num_hidden_layers: 1,
        num_attention_heads: 16,
        num_key_value_heads: 8,
        head_dim: 128,
        max_position_embeddings: 512,
        attention_bias: false,
        rms_norm_eps: 1e-6,
        rope_theta: 1e6,
        hidden_act: Activation::Silu,
    }
}

fn check_mlp(device: &Device) -> Result<()> {
    use std::collections::HashMap;
    let cfg = mlp_config();
    let gate = candle_nn::Linear::new(Tensor::rand(-0.05f32, 0.05, (3072, 1024), device)?, None);
    let up = candle_nn::Linear::new(Tensor::rand(-0.05f32, 0.05, (3072, 1024), device)?, None);
    let down = candle_nn::Linear::new(Tensor::rand(-0.05f32, 0.05, (1024, 3072), device)?, None);
    let weights = HashMap::from([
        ("gate_proj.weight".to_owned(), gate.weight().clone()),
        ("up_proj.weight".to_owned(), up.weight().clone()),
        ("down_proj.weight".to_owned(), down.weight().clone()),
    ]);
    let vb = VarBuilder::from_tensors(weights, DType::F32, device);
    let model = Stage0Qwen3Mlp::load(&cfg, vb.clone())?;
    assert_eq!(
        model.gate_up_proj.is_some(),
        device.is_cpu() && candle_core::utils::has_accelerate()
    );
    let bits = |t: Tensor| -> candle_core::Result<Vec<u32>> {
        Ok(t.flatten_all()?
            .to_vec1::<f32>()?
            .into_iter()
            .map(f32::to_bits)
            .collect())
    };
    for batch in [1, 2] {
        for seq in [1, 7, 8, 33, 157] {
            for strided in [false, true] {
                let xs = if strided {
                    Tensor::rand(-1f32, 1., (batch, 1024, seq), device)?.transpose(1, 2)?
                } else {
                    Tensor::rand(-1f32, 1., (batch, seq, 1024), device)?
                };
                let lhs = gate.forward(&xs)?.silu()?;
                let expected = bits(down.forward(&(lhs * up.forward(&xs)?)?)?)?;
                for _ in 0..2 {
                    assert_eq!(
                        bits(model.forward(&xs)?)?,
                        expected,
                        "batch={batch} seq={seq} strided={strided}"
                    );
                }
            }
        }
    }
    for xs in [
        Tensor::ones((8, 1024), DType::F32, device)?,
        Tensor::ones((1, 2, 8, 1024), DType::F32, device)?,
    ] {
        let expected = down.forward(&(gate.forward(&xs)?.silu()? * up.forward(&xs)?)?)?;
        assert_eq!(bits(model.forward(&xs)?)?, bits(expected)?);
    }
    let mut other_act = cfg;
    other_act.hidden_act = Activation::Relu;
    assert!(Stage0Qwen3Mlp::load(&other_act, vb)?.gate_up_proj.is_none());
    Ok(())
}

#[test]
fn cpu_merged_mlp_preserves_bits_and_fallbacks() -> Result<()> {
    check_mlp(&Device::Cpu)
}

#[cfg(all(feature = "metal", target_os = "macos"))]
#[test]
fn metal_mlp_keeps_original_route_and_bits() -> Result<()> {
    check_mlp(&Device::new_metal(0)?)
}

#[test]
fn cpu_mlp_keeps_variable_weights_live() -> Result<()> {
    let mut vars = candle_nn::VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &Device::Cpu);
    let model = Stage0Qwen3Mlp::load(&mlp_config(), vb)?;
    assert!(model.gate_up_proj.is_none());
    vars.set_one(
        "gate_proj.weight",
        Tensor::zeros((3072, 1024), DType::F32, &Device::Cpu)?,
    )?;
    let xs = Tensor::ones((1, 8, 1024), DType::F32, &Device::Cpu)?;
    assert!(
        model
            .forward(&xs)?
            .flatten_all()?
            .to_vec1::<f32>()?
            .iter()
            .all(|x| *x == 0.)
    );
    Ok(())
}

#[test]
fn cpu_mlp_keeps_strided_weights_on_original_route() -> Result<()> {
    use std::collections::HashMap;
    let device = Device::Cpu;
    let gate = candle_nn::Linear::new(
        Tensor::rand(-0.05f32, 0.05, (1024, 3072), &device)?.t()?,
        None,
    );
    let up = candle_nn::Linear::new(
        Tensor::rand(-0.05f32, 0.05, (1024, 3072), &device)?.t()?,
        None,
    );
    let down = candle_nn::Linear::new(Tensor::rand(-0.05f32, 0.05, (1024, 3072), &device)?, None);
    let vb = VarBuilder::from_tensors(
        HashMap::from([
            ("gate_proj.weight".to_owned(), gate.weight().clone()),
            ("up_proj.weight".to_owned(), up.weight().clone()),
            ("down_proj.weight".to_owned(), down.weight().clone()),
        ]),
        DType::F32,
        &device,
    );
    let model = Stage0Qwen3Mlp::load(&mlp_config(), vb)?;
    assert!(model.gate_up_proj.is_none());
    for seq in [1, 8] {
        let xs = Tensor::rand(-1f32, 1., (1, seq, 1024), &device)?;
        let expected = down.forward(&(gate.forward(&xs)?.silu()? * up.forward(&xs)?)?)?;
        let bits = |t: Tensor| -> candle_core::Result<Vec<u32>> {
            Ok(t.flatten_all()?
                .to_vec1::<f32>()?
                .into_iter()
                .map(f32::to_bits)
                .collect())
        };
        assert_eq!(bits(model.forward(&xs)?)?, bits(expected)?);
    }
    Ok(())
}

#[test]
fn cpu_grouped_attention_matches_repeated_kv_bits() -> Result<()> {
    let device = Device::Cpu;
    for batch in [1, 2] {
        for seq in [1, 7, 8, 33, 157, 179, 287] {
            for dim in [8, 128] {
                let q = Tensor::rand(-1f32, 1.0, (batch, 4, seq, dim), &device)?;
                let k = Tensor::rand(-1f32, 1.0, (batch, 2, seq, dim), &device)?;
                let v = Tensor::rand(-1f32, 1.0, (batch, seq, 2, dim), &device)?.transpose(1, 2)?;
                let mask = Tensor::from_vec(
                    (0..batch * seq * seq)
                        .map(|i| {
                            if seq > 1 && i % seq == seq - 1 {
                                f32::NEG_INFINITY
                            } else {
                                0.
                            }
                        })
                        .collect::<Vec<_>>(),
                    (batch, 1, seq, seq),
                    &device,
                )?;
                let repeated_k = candle_transformers::utils::repeat_kv(k.clone(), 2)?;
                let repeated_v = candle_transformers::utils::repeat_kv(v.clone(), 2)?;
                let strided_k = k.transpose(2, 3)?.contiguous()?.transpose(2, 3)?;
                let scale = 1. / (dim as f64).sqrt();
                for mask in [None, Some(&mask)] {
                    let expected = attention_manual(&q, &repeated_k, &repeated_v, mask, scale)?;
                    for _ in 0..2 {
                        let actual = cpu_grouped_attention(&q, &k, &v, mask, scale)?;
                        if dim != 128 || seq < 8 || !candle_core::utils::has_accelerate() {
                            assert!(actual.is_none());
                        } else {
                            let actual = actual.expect("eligible CPU GQA");
                            let expected = expected.flatten_all()?.to_vec1::<f32>()?;
                            let actual = actual.flatten_all()?.to_vec1::<f32>()?;
                            let differing = actual
                                .iter()
                                .zip(&expected)
                                .filter(|(a, b)| a.to_bits() != b.to_bits())
                                .count();
                            assert_eq!(differing, 0, "batch={batch} seq={seq} dim={dim}");
                        }
                    }
                    if seq > 1 {
                        assert!(cpu_grouped_attention(&q, &strided_k, &v, mask, scale)?.is_none());
                        let strided_q = q.transpose(2, 3)?.contiguous()?.transpose(2, 3)?;
                        assert!(cpu_grouped_attention(&strided_q, &k, &v, mask, scale)?.is_none());
                    }
                    assert!(
                        cpu_grouped_attention(&q, &repeated_k, &repeated_v, mask, scale)?.is_none()
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
fn cpu_attention_transposed_key_matches_materialized_bits() -> Result<()> {
    let device = Device::Cpu;
    for batch in [1, 2] {
        for seq in [1, 7, 33, 157] {
            for dim in [8, 128] {
                let q = Tensor::rand(-0.5f32, 0.5, (batch, 4, seq, dim), &device)?;
                let k = Tensor::rand(-0.5f32, 0.5, (batch, 4, seq, dim), &device)?;
                let v = Tensor::rand(-0.5f32, 0.5, (batch, 4, seq, dim), &device)?;
                let strided_k = k.transpose(2, 3)?.contiguous()?.transpose(2, 3)?;
                let mask = Tensor::from_vec(
                    (0..batch * seq * seq)
                        .map(|i| {
                            if seq > 1 && i % seq == seq - 1 {
                                f32::NEG_INFINITY
                            } else {
                                0f32
                            }
                        })
                        .collect::<Vec<_>>(),
                    (batch, 1, seq, seq),
                    &device,
                )?;
                for k in [&k, &strided_k] {
                    for mask in [None, Some(&mask)] {
                        let scale = 1.0 / (dim as f64).sqrt();
                        let mut scores = q
                            .matmul(&k.transpose(2, 3)?.contiguous()?)?
                            .affine(scale, 0.0)?;
                        if let Some(mask) = mask {
                            scores = scores.broadcast_add(mask)?;
                        }
                        let expected = candle_nn::ops::softmax_last_dim(&scores)?
                            .matmul(&v)?
                            .flatten_all()?
                            .to_vec1::<f32>()?
                            .into_iter()
                            .map(f32::to_bits)
                            .collect::<Vec<_>>();
                        for _ in 0..2 {
                            let actual = attention_manual(&q, k, &v, mask, scale)?
                                .flatten_all()?
                                .to_vec1::<f32>()?
                                .into_iter()
                                .map(f32::to_bits)
                                .collect::<Vec<_>>();
                            assert_eq!(actual, expected, "batch={batch} seq={seq} dim={dim}");
                        }
                    }
                }
            }
        }
    }
    Ok(())
}
