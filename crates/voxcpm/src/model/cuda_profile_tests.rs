//! Explicit local attribution; synchronized timings do not measure graph throughput.
use super::*;

#[test]
#[ignore = "local original BF16 weights and CUDA; no model downloads"]
fn real_bf16_module_attribution() -> anyhow::Result<()> {
    let directory = std::env::var_os("TALECHIME_VOX_PROFILE_MODEL").expect("model required");
    let text = std::fs::read_to_string(
        std::env::var_os("TALECHIME_VOX_PROFILE_TEXT").expect("text required"),
    )?;
    let output = std::env::var_os("TALECHIME_VOX_PROFILE_OUTPUT").expect("output required");
    anyhow::ensure!(
        std::env::var_os("VOXCPM_CUDA_GRAPH").is_none(),
        "baseline must use the default graph setting"
    );
    let device = Device::new_cuda(0)?;
    let mut model = Model::load_original(Path::new(&directory), &device, DType::BF16, &|| false)?;
    let mut baseline = None;
    let mut rows = Vec::new();
    for (index, measure_modules) in [false, false, true, true].into_iter().enumerate() {
        let mut pcm_bits = Vec::new();
        let mut first_pcm = None;
        let mut boundary = talechime_core::audio::BoundarySilence::default();
        let mut boundary_samples = 0;
        device.synchronize()?;
        let started = std::time::Instant::now();
        let outcome = model.generate(
            text.trim(),
            None,
            &Options {
                seed: talechime_core::SeedPolicy::Pinned(42).resolve(0, 0),
                max_frames: 187,
                measure_modules,
                ..Default::default()
            },
            &|| false,
            |pcm| {
                first_pcm.get_or_insert(started.elapsed().as_secs_f64());
                if pcm.iter().any(|v| !v.is_finite()) {
                    candle_core::bail!("nonfinite diagnostic PCM");
                }
                pcm_bits.extend(pcm.iter().map(|v| v.to_bits()));
                let filtered = boundary
                    .push(talechime_core::backend::Pcm {
                        samples: pcm.to_vec(),
                        sample_rate: SAMPLE_RATE,
                        channels: 1,
                    })
                    .map_err(|e| candle_core::Error::Msg(e.to_string()))?;
                if let Some(filtered) = filtered {
                    boundary_samples += filtered.samples.len();
                }
                Ok(())
            },
        )?;
        device.synchronize()?;
        let seconds = started.elapsed().as_secs_f64();
        if let Some(tail) = boundary.finish(true)? {
            boundary_samples += tail.samples.len();
        }
        anyhow::ensure!(outcome == Outcome::Eos && !pcm_bits.is_empty());
        if let Some(expected) = &baseline {
            anyhow::ensure!(&pcm_bits == expected, "diagnostic changed raw PCM bits");
        } else {
            baseline = Some(pcm_bits.clone());
        }
        let timings = model.timings();
        let graph = model.cuda_graph_status();
        rows.push(serde_json::json!({
            "index": index,
            "measure_modules": measure_modules,
            "graph_launches_disabled_by_module_timing": measure_modules,
            "cached_cfm_graph_present": graph.cfm,
            "cached_decoder_graph_present": graph.decoder,
            "seconds": seconds,
            "first_pcm_seconds": first_pcm,
            "raw_pcm_samples": pcm_bits.len(),
            "core_boundary_silence_samples": boundary_samples,
            "pcm_bits_equal_default": true,
            "natural_eos": true,
            "stages_seconds": {
                "prefill": timings.prefill.as_secs_f64(),
                "cfm": timings.cfm.as_secs_f64(),
                "decoder": timings.decoder.as_secs_f64(),
                "local_encoder": timings.local_encoder.as_secs_f64(),
                "backbone": timings.backbone.as_secs_f64(),
                "residual": timings.residual.as_secs_f64()
            }
        }));
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(
        serde_json::to_string_pretty(&serde_json::json!({
            "kind": "module_attribution_not_production_performance",
            "scope": "BF16 original narrator; one owner; two default then two synchronized eager requests",
            "source_text_sha256": talechime_protocol::text_hash(&text),
            "rows": rows
        }))?
        .as_bytes(),
    )?;
    Ok(())
}
