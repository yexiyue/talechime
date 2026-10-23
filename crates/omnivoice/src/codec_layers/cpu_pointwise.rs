//! Offline real-activation screen. The candidate is not a production switch.
use std::{hint::black_box, time::Instant};

use super::*;
use crate::cpu_attribution as io;

fn snake(xs: &Tensor, alpha: &Tensor) -> CandleResult<Tensor> {
    let sin = alpha.broadcast_mul(xs)?.sin()?;
    xs + (alpha + 1e-9)?.recip()?.broadcast_mul(&(&sin * &sin)?)?
}

#[test]
#[ignore = "requires immutable local weights and prior real decoder activation capture"]
fn compare_real_pointwise() -> Result<()> {
    assert!(candle_core::utils::has_accelerate());
    let bundle = crate::stage1_decoder::Stage1DecoderBundle::from_model_root(io::model()?)?;
    let ws = CodecWeightSource::from_weight_path(bundle.weights_path(), DType::F32, &Device::Cpu)?;
    let mut channels = bundle.model_config().decoder_hidden_dim;
    for block in 0..bundle.model_config().decoder_strides.len() {
        channels /= 2;
        let base = format!("decoder-frames-130-block-{block}");
        let capture = io::load(&format!("{base}-res_unit1-conv2"))?;
        let residual = io::load(&format!("{base}-res_unit1"))?;
        let xs = &capture["input"];
        let residual = &residual["input"];
        let weights = ws.pp(format!("acoustic_decoder.block.{block}"));
        let conv = weights
            .pp("res_unit1.conv2")
            .load_conv1d_flexible(channels, channels, 1, 1, 0)?;
        let CodecConv1dInner::Dense(layer) = &conv.inner else {
            return Err(crate::OmniVoiceError::InvalidData(
                "dense fixture required".into(),
            ));
        };
        let alpha = weights
            .pp("res_unit2.snake1")
            .get((1, channels, 1), "alpha")?;
        let reference = layer.forward(xs)?;
        io::equal(&reference, &capture["output"])?;
        io::equal(&reference, &conv.forward_decoder_pointwise(xs)?)?;
        let expected = snake(&(&reference + residual)?, &alpha)?;
        io::equal(
            &expected,
            &snake(&(conv.forward_decoder_pointwise(xs)? + residual)?, &alpha)?,
        )?;
        for boundary in ["conv_bias", "residual_next_snake"] {
            for (slot, variant) in ["original", "retained", "retained", "original"]
                .iter()
                .enumerate()
            {
                let run = || -> CandleResult<Tensor> {
                    let result = if *variant == "original" {
                        layer.forward(xs)?
                    } else {
                        conv.forward_decoder_pointwise(xs)?
                    };
                    if boundary == "conv_bias" {
                        Ok(result)
                    } else {
                        snake(&(result + residual)?, &alpha)
                    }
                };
                black_box(run()?);
                for group in 0..5 {
                    let start = Instant::now();
                    for _ in 0..5 {
                        black_box(run()?);
                    }
                    println!(
                        "CPU_TRACE {}",
                        serde_json::json!({"case":base,"boundary":boundary,
                        "slot":slot,"variant":variant,"group":group,"microseconds":start.elapsed().as_secs_f64()*1e6/5.0,
                        "input_shape":xs.dims(),"bits_equal":true})
                    );
                }
            }
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires local weights, fixed codes and original full decoder output archives"]
fn compare_fixed_decoder_bits() -> Result<()> {
    let bundle = crate::stage1_decoder::Stage1DecoderBundle::from_model_root(io::model()?)?;
    let ws = CodecWeightSource::from_weight_path(bundle.weights_path(), DType::F32, &Device::Cpu)?;
    let model = crate::stage1_model::Stage1Model::load(&ws, bundle.model_config())?;
    let mut files = std::fs::read_dir(io::root()?.join("codes"))?
        .map(|entry| entry.map(|entry| entry.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.sort();
    for path in files {
        let codes: serde_json::Value = serde_json::from_slice(&std::fs::read(&path)?)?;
        let codebooks = codes["codebooks"]
            .as_u64()
            .ok_or_else(|| crate::OmniVoiceError::InvalidData("missing codebooks".into()))?
            as usize;
        let frames = codes["frames"]
            .as_u64()
            .ok_or_else(|| crate::OmniVoiceError::InvalidData("missing frames".into()))?
            as usize;
        let values: Vec<i64> = serde_json::from_value(codes["values"].clone())?;
        let tokens = crate::contracts::I64Tensor2::new((codebooks, frames), values)?;
        let name = path
            .file_stem()
            .and_then(|v| v.to_str())
            .ok_or_else(|| crate::OmniVoiceError::InvalidData("invalid fixture name".into()))?;
        let reference = io::load(&format!("decoder-{name}-raw"))?;
        for repeat in 0..2 {
            let output = model.decode_tokens(&tokens, &Device::Cpu)?;
            io::equal(&output, &reference["output"])?;
            println!(
                "CPU_TRACE {}",
                serde_json::json!({"case":name,"repeat":repeat,
                "bits_equal":true,"samples":output.elem_count()})
            );
        }
    }
    Ok(())
}
