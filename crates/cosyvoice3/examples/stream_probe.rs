//! Chain-A streaming parity: fixed official tokens -> Rust streaming
//! flow/HiFT, compared chunk-by-chunk against the official stream mode
//! (cosyvoice3_stream_reference.py exports).
//!
//! Usage: `stream_probe <model-dir> <fixture-dir> [metal]`
use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use std::path::Path;
use std::time::Instant;
use talechime_cosyvoice3::{
    CausalHiFtGenerator, CausalMaskedDiffWithDiT, CosyVoice3Config, DiT, TokenStreamDecoder,
};

fn load_weights(path: &Path, dev: &Device) -> Result<VarBuilder<'static>> {
    Ok(unsafe { VarBuilder::from_mmaped_safetensors(&[path], DType::F32, dev)? })
}

fn stats(a: &[f32], b: &[f32]) -> Result<(f32, f32, f64)> {
    anyhow::ensure!(!a.is_empty(), "empty comparison");
    anyhow::ensure!(
        a.len() == b.len(),
        "sample length mismatch: {} != {}",
        a.len(),
        b.len()
    );
    let mut max = 0f32;
    let mut se = 0f64;
    let mut ps = 0f64;
    for (&a, &b) in a.iter().zip(b) {
        anyhow::ensure!(
            a.is_finite() && b.is_finite(),
            "non-finite comparison input"
        );
        let d = f64::from(a) - f64::from(b);
        max = max.max(d.abs() as f32);
        se += d * d;
        ps += f64::from(b).powi(2);
    }
    let rmse = (se / a.len() as f64).sqrt();
    let snr = 10.0 * (ps / se.max(1e-30)).log10();
    Ok((max, rmse as f32, snr))
}

fn reference_chunk_names(keys: impl Iterator<Item = String>) -> Result<Vec<String>> {
    let keys: std::collections::BTreeSet<_> = keys.collect();
    anyhow::ensure!(keys.contains("final"), "missing official final chunk");
    let count = keys.iter().filter(|k| k.starts_with("chunk_")).count();
    let mut names = Vec::with_capacity(count + 1);
    for i in 0..count {
        let name = format!("chunk_{i:02}");
        anyhow::ensure!(keys.contains(&name), "missing official {name}");
        names.push(name);
    }
    names.push("final".to_string());
    let mel_count = keys.iter().filter(|k| k.starts_with("flow_mel_")).count();
    anyhow::ensure!(
        mel_count == names.len(),
        "official mel/PCM chunk count mismatch"
    );
    for i in 0..names.len() {
        anyhow::ensure!(
            keys.contains(&format!("flow_mel_{i:02}")),
            "missing official mel chunk {i}"
        );
    }
    Ok(names)
}

fn main() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let model = Path::new(
        args.get(1)
            .context("usage: stream_probe <model> <fixture> [metal]")?,
    );
    let dir = Path::new(args.get(2).context("missing fixture dir")?);
    let dev = match args.get(3).map(String::as_str) {
        Some("metal") => Device::new_metal(0)?,
        _ => Device::Cpu,
    };

    let cfg = CosyVoice3Config::from_model_dir(model)?;
    let load = |p: std::path::PathBuf| -> Result<std::collections::HashMap<String, Tensor>> {
        let mut out = std::collections::HashMap::new();
        for (k, t) in candle_core::safetensors::load(p, &Device::Cpu)? {
            // Metal has no integer dtype casts: convert while still on CPU.
            let t = if matches!(t.dtype(), DType::I32 | DType::I64) {
                t.to_dtype(DType::U32)?
            } else {
                t
            };
            out.insert(k, t.to_device(&dev)?);
        }
        Ok(out)
    };
    let front = load(dir.join("frontend.safetensors"))?;
    let llm_fx = load(dir.join("llm.safetensors"))?;
    let stream = load(dir.join("stream/stream-chunks.safetensors"))?;
    let hift_fx = load(dir.join("hift.safetensors"))?;
    let expected_chunks = reference_chunk_names(stream.keys().cloned())?;

    let vb = load_weights(&model.join("flow.safetensors"), &dev)?;
    let dit = DiT::new(cfg.flow.dit.clone(), vb.pp("dit"))?;
    let flow = CausalMaskedDiffWithDiT::new(&cfg.flow, 192, dit, vb)?;
    let mut hift = CausalHiFtGenerator::new(
        cfg.hift,
        0,
        load_weights(&model.join("hift.safetensors"), &dev)?,
    )?;
    // Share the official session's source buffers.
    hift.set_source_noise(stream["rand_ini"].clone(), stream["source_noise"].clone())?;

    let tokens: Vec<u32> = llm_fx["speech_tokens"]
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?
        .iter()
        .map(|&v| v as u32)
        .collect();

    let mut dec = TokenStreamDecoder::new(
        flow,
        hift,
        &front["prompt_speech_token"].to_dtype(DType::U32)?,
        &front["prompt_speech_feat"],
        &front["prompt_embedding"],
    )?;
    dec.extend_tokens(&tokens)?;

    let mut rust_all: Vec<f32> = Vec::new();
    let mut failures = 0usize;
    let t0 = Instant::now();
    let mut first_chunk_at = None;
    let mut mel_offset = 0;
    let mut official_stream = Vec::new();
    for (chunk_idx, name) in expected_chunks.iter().enumerate() {
        let started = Instant::now();
        let chunk = dec
            .next_pcm(true)?
            .context("Rust stream ended before official chunks")?;
        let elapsed = started.elapsed().as_secs_f64();
        first_chunk_at.get_or_insert_with(|| t0.elapsed().as_secs_f64());
        anyhow::ensure!(
            chunk.final_chunk == (name == "final"),
            "{name}: final flag mismatch"
        );
        anyhow::ensure!(!chunk.hit_token_limit, "{name}: unexpected token limit");
        let bv = stream[name]
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let (max, rmse, snr) =
            stats(&chunk.samples, &bv).with_context(|| format!("{name}: PCM"))?;
        let m = dec
            .last_new_mel()
            .context("missing Rust mel chunk")?
            .to_dtype(DType::F32)?;
        let om = &stream[&format!("flow_mel_{chunk_idx:02}")];
        anyhow::ensure!(
            om.dim(2)? >= mel_offset,
            "{name}: official mel moved backwards"
        );
        let expected_mel = om.narrow(2, mel_offset, om.dim(2)? - mel_offset)?;
        anyhow::ensure!(
            m.dims() == expected_mel.dims(),
            "{name}: mel shape {:?} != {:?}",
            m.dims(),
            expected_mel.dims()
        );
        let mv = m.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
        let ov = expected_mel
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let (mmax, mrmse, _) = stats(&mv, &ov).with_context(|| format!("{name}: mel"))?;
        mel_offset = om.dim(2)?;
        candle_core::safetensors::save(
            &std::iter::once(("mel".to_string(), m.to_device(&Device::Cpu)?)).collect(),
            dir.join(format!("stream/rust_mel_{chunk_idx:02}.safetensors")),
        )?;
        println!(
            "{name}: samples={} max_abs={max:.3e} rmse={rmse:.3e} snr_db={snr:.2} decode_s={elapsed:.3} mel_max={mmax:.3e} mel_rmse={mrmse:.3e}",
            chunk.samples.len()
        );
        // Preserve the existing streaming gates; these do not establish
        // strict waveform parity (compare_model retains its tighter gate).
        if mrmse > 1e-3 || rmse > 1e-1 {
            failures += 1;
            println!("{name}: GATE FAILED (mel_rmse={mrmse:.3e} pcm_rmse={rmse:.3e})");
        }
        rust_all.extend_from_slice(&chunk.samples);
        official_stream.extend_from_slice(&bv);
    }
    anyhow::ensure!(dec.next_pcm(true)?.is_none(), "unexpected extra Rust chunk");
    let (max, rmse, snr) = stats(&rust_all, &official_stream)?;
    println!(
        "whole_stream: samples={} vs {} max_abs={max:.3e} rmse={rmse:.3e} snr_db={snr:.2}",
        rust_all.len(),
        official_stream.len()
    );
    let ref_pcm = hift_fx["pcm"]
        .to_dtype(DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let (gap_max, gap_rmse, _) = stats(&rust_all, &ref_pcm)?;
    println!(
        "rust_stream_vs_official_nonstream: max_abs={gap_max:.3e} rmse={gap_rmse:.3e} (official's own gap: see stream/meta.json)"
    );
    if let Some(t) = first_chunk_at {
        println!(
            "first_chunk_latency_s={t:.3} total_s={:.3} audio_s={:.3}",
            t0.elapsed().as_secs_f64(),
            rust_all.len() as f64 / 24000.0
        );
    }
    anyhow::ensure!(failures == 0, "{failures} stream chunk(s) failed parity");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_rejects_missing_samples_and_nonfinite_values() {
        assert!(stats(&[], &[]).is_err());
        assert!(stats(&[0.0], &[0.0, 1.0]).is_err());
        assert!(stats(&[f32::NAN], &[0.0]).is_err());
        assert!(stats(&[0.0], &[f32::INFINITY]).is_err());
        assert_eq!(stats(&[1.0, 2.0], &[1.0, 2.0]).unwrap().0, 0.0);
    }

    #[test]
    fn reference_requires_contiguous_chunks_final_and_all_mels() {
        let names = |keys: &[&str]| reference_chunk_names(keys.iter().map(|s| s.to_string()));
        assert!(names(&["chunk_00", "flow_mel_00"]).is_err());
        assert!(names(&["chunk_01", "final", "flow_mel_00", "flow_mel_01"]).is_err());
        assert!(names(&["chunk_00", "final", "flow_mel_00"]).is_err());
        assert!(names(&["final", "flow_mel_01"]).is_err());
        assert_eq!(
            names(&["chunk_00", "final", "flow_mel_00", "flow_mel_01"]).unwrap(),
            ["chunk_00", "final"]
        );
    }
}
