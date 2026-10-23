//! Compare padded transposed convolution with zero-padding plus output cropping.
use std::{error::Error, path::PathBuf, time::Instant};

use candle_core::{DType, Tensor};
use candle_nn::{ConvTranspose1d, ConvTranspose1dConfig, Module, VarBuilder};
use talechime_omnivoice::{DeviceSpec, stage1_decoder::Stage1DecoderBundle};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: conv_transpose_probe MODEL cpu|metal [FRAMES]".into());
    }
    let root = PathBuf::from(&args[0]);
    let device = match args[1].as_str() {
        "cpu" => DeviceSpec::Cpu.resolve()?,
        "metal" => DeviceSpec::Metal.resolve()?,
        _ => return Err("device must be cpu or metal".into()),
    };
    let bundle = Stage1DecoderBundle::from_model_root(&root)?;
    let cfg = bundle.model_config();
    // SAFETY: the local pinned model file remains immutable during this probe.
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[bundle.weights_path()], DType::F32, &device)?
    };
    let mut frames: usize = args.get(2).map_or(Ok(130), |s| s.parse())?;
    if !(1..=1024).contains(&frames) {
        return Err("input frames must be in 1..=1024".into());
    }
    let mut channels = cfg.decoder_hidden_dim;
    for (index, &stride) in cfg.decoder_strides.iter().enumerate() {
        let out_channels = channels / 2;
        let layer_vb = vb.pp(format!("acoustic_decoder.block.{index}.conv_t1"));
        let weight = layer_vb.get((channels, out_channels, 2 * stride), "weight")?;
        let bias = layer_vb.get(out_channels, "bias")?;
        let config = ConvTranspose1dConfig {
            stride,
            padding: stride.div_ceil(2),
            output_padding: stride % 2,
            ..Default::default()
        };
        let original = ConvTranspose1d::new(weight.clone(), Some(bias.clone()), config);
        let input = (0..channels * frames)
            .map(|i| (i as f32 * 0.013).sin() * 0.1)
            .collect::<Vec<_>>();
        let xs = Tensor::from_vec(input, (1, channels, frames), &device)?;
        let cropped = || -> candle_core::Result<Tensor> {
            let full = xs.conv_transpose1d(&weight, 0, 0, stride, 1, 1)?;
            let len = full.dim(2)? - 2 * config.padding + config.output_padding;
            full.narrow(2, config.padding, len)?
                .broadcast_add(&bias.reshape((1, out_channels, 1))?)
        };
        let reference = original.forward(&xs)?;
        let candidate = cropped()?;
        device.synchronize()?;
        let a = reference.flatten_all()?.to_vec1::<f32>()?;
        let b = candidate.flatten_all()?.to_vec1::<f32>()?;
        if a.len() != b.len() || b.iter().any(|v| !v.is_finite()) {
            return Err("invalid candidate output".into());
        }
        let mut max_abs = 0f64;
        let mut diff_sq = 0f64;
        let mut ref_sq = 0f64;
        for (&a, &b) in a.iter().zip(&b) {
            let diff = f64::from(a) - f64::from(b);
            max_abs = max_abs.max(diff.abs());
            diff_sq += diff * diff;
            ref_sq += f64::from(a).powi(2);
        }
        let relative_rms = (diff_sq / ref_sq.max(1e-30)).sqrt();
        if relative_rms > 1e-4 {
            return Err(format!("block {index}: relative RMS error {relative_rms}").into());
        }
        let mut original_ms = Vec::new();
        let mut cropped_ms = Vec::new();
        for _ in 0..5 {
            device.synchronize()?;
            let t = Instant::now();
            let _out = original.forward(&xs)?;
            device.synchronize()?;
            original_ms.push(t.elapsed().as_secs_f64() * 1000.0);
            let t = Instant::now();
            let _out = cropped()?;
            device.synchronize()?;
            cropped_ms.push(t.elapsed().as_secs_f64() * 1000.0);
        }
        println!(
            "{}",
            serde_json::json!({
                "block": index, "device": args[1], "input_frames": frames,
                "in_channels": channels, "out_channels": out_channels,
                "stride": stride, "padding": config.padding,
                "output_padding": config.output_padding,
                "original_ms": original_ms, "cropped_ms": cropped_ms,
                "max_abs": max_abs, "relative_rms": relative_rms,
            })
        );
        frames *= stride;
        channels = out_channels;
    }
    Ok(())
}
