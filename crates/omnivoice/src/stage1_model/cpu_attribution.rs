//! Offline decoder attribution using real codes and activation inputs.
use std::{fs, time::Instant};

use super::*;
use crate::cpu_attribution as io;
use candle_core::{DType, Device};
use serde::Deserialize;

#[derive(Deserialize)]
struct Codes {
    codebooks: usize,
    frames: usize,
    values: Vec<i64>,
}

fn measure(
    name: &str,
    op: &str,
    xs: &Tensor,
    export: bool,
    mut run: impl FnMut(&Tensor) -> Result<Tensor>,
) -> Result<Tensor> {
    let output = run(xs)?;
    if export {
        io::save(name, &[("input", xs), ("output", &output)])?;
    }
    io::bench(name, op, || run(xs))?;
    Ok(output)
}

#[test]
#[ignore = "requires local decoder weights and a directory of fixed code JSON files"]
fn replay_real_cpu_decoder() -> Result<()> {
    let bundle = crate::stage1_decoder::Stage1DecoderBundle::from_model_root(io::model()?)?;
    let started = Instant::now();
    let ws = CodecWeightSource::from_weight_path(bundle.weights_path(), DType::F32, &Device::Cpu)?;
    let model = Stage1Model::load(&ws, bundle.model_config())?;
    println!(
        "CPU_TRACE {}",
        serde_json::json!({"kind":"load","seconds":started.elapsed().as_secs_f64()})
    );
    let code_root = io::root()?.join("codes");
    let mut files = fs::read_dir(code_root)?
        .map(|e| e.map(|e| e.path()))
        .collect::<std::io::Result<Vec<_>>>()?;
    files.sort();
    let export = std::env::var_os("OMNI_CPU_TRACE_EXPORT").is_some();
    let mut selected = None;
    for path in files {
        let codes: Codes = serde_json::from_slice(&fs::read(&path)?)?;
        let tokens = I64Tensor2::new((codes.codebooks, codes.frames), codes.values)?;
        let name = path
            .file_stem()
            .and_then(|n| n.to_str())
            .ok_or_else(|| OmniVoiceError::InvalidData("invalid code filename".into()))?;
        let started = Instant::now();
        let raw = model.decode_tokens(&tokens, &Device::Cpu)?;
        println!(
            "CPU_TRACE {}",
            serde_json::json!({"case":name,"op":"first_decode","microseconds":started.elapsed().as_secs_f64()*1e6,"samples":raw.elem_count()})
        );
        // Full decoding uses production code; the traversal below is independently checked.
        for group in 0..5 {
            let started = Instant::now();
            let output = model.decode_tokens(&tokens, &Device::Cpu)?;
            let elapsed = started.elapsed().as_secs_f64() * 1e6;
            io::equal(&raw, &output)?;
            println!(
                "CPU_TRACE {}",
                serde_json::json!({"case":name,"op":"decode","group":group,"microseconds":elapsed,"samples":raw.elem_count()})
            );
        }
        if export {
            io::save(&format!("decoder-{name}-raw"), &[("output", &raw)])?;
        }
        if name == "frames-130" {
            selected = Some(tokens);
        }
    }
    if let Some(tokens) = selected {
        trace_decoder(&model, &tokens, export)?;
    }
    Ok(())
}

fn trace_decoder(model: &Stage1Model, tokens: &I64Tensor2, export: bool) -> Result<()> {
    let name = "frames-130";
    let raw = model.decode_tokens(tokens, &Device::Cpu)?;
    let trace = model.decode_tokens_with_trace(tokens, &Device::Cpu)?;
    io::equal(&raw, &trace.raw_waveform)?;
    let dec = &model.decoder;
    let mut x = measure(
        &format!("decoder-{name}-conv1"),
        "conv7",
        &trace.decoder_input,
        export,
        |x| Ok(x.apply(&dec.conv1)?),
    )?;
    for (index, block) in dec.blocks.iter().enumerate() {
        let base = format!("decoder-{name}-block-{index}");
        let block_out = measure(&base, "block", &x, export, |x| Ok(x.apply(block)?))?;
        let snake = measure(&format!("{base}-snake1"), "snake", &x, export, |x| {
            Ok(x.apply(&block.snake1)?)
        })?;
        x = measure(&format!("{base}-conv_t1"), "deconv", &snake, export, |x| {
            Ok(x.apply(&block.conv_t1)?)
        })?;
        for (unit_index, unit) in [&block.res_unit1, &block.res_unit2, &block.res_unit3]
            .iter()
            .enumerate()
        {
            let unit_base = format!("{base}-res_unit{}", unit_index + 1);
            let unit_out = measure(&unit_base, "residual_consumer", &x, export, |x| {
                Ok(x.apply(*unit)?)
            })?;
            // Export representative real primitive inputs once per block; all units still get consumer timing.
            if unit_index == 0 {
                let y = x.apply(&unit.snake1)?;
                let y = measure(&format!("{unit_base}-conv1"), "conv7", &y, export, |x| {
                    Ok(x.apply(&unit.conv1)?)
                })?;
                let y = y.apply(&unit.snake2)?;
                let y = measure(&format!("{unit_base}-conv2"), "conv1", &y, export, |x| {
                    Ok(unit.conv2.forward_decoder_pointwise(x)?)
                })?;
                io::equal(&(&y + &x)?, &unit_out)?;
                io::bench(&unit_base, "layout_materialize", || {
                    Ok(x.transpose(1, 2)?
                        .contiguous()?
                        .transpose(1, 2)?
                        .contiguous()?)
                })?;
            }
            x = unit_out;
        }
        io::equal(&x, &block_out)?;
    }
    let output = x.apply(&dec.snake1)?.apply(&dec.conv2)?;
    io::equal(&output, &raw)?;
    Ok(())
}
