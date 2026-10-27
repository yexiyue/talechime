//! Model arithmetic gate; production Backend ownership is checked separately.
use super::*;

#[test]
#[ignore = "local original BF16 model and CUDA; no downloads"]
fn real_bf16_decoder_graph_matches_eager_full_request() -> anyhow::Result<()> {
    let root = std::env::var_os("TALECHIME_VOX_F32_GRAPH_MODEL").expect("model required");
    let text = std::fs::read_to_string(
        std::env::var_os("TALECHIME_VOX_F32_GRAPH_TEXT").expect("text required"),
    )?;
    let output = std::env::var_os("TALECHIME_VOX_F32_GRAPH_OUTPUT").expect("output required");
    anyhow::ensure!(std::env::var_os("VOXCPM_CUDA_GRAPH").is_none());
    let device = Device::new_cuda(0)?;
    let mut model = Model::load_original(Path::new(&root), &device, DType::BF16, &|| false)?;
    let mut baseline = None;
    let mut rows = Vec::new();
    for graph_enabled in [false, false, true, true, false] {
        if !graph_enabled {
            model.decoder_graph = None;
        } else if model.decoder_graph.is_none() {
            model.decoder_graph = Some(crate::cuda_graph::decoder::DecoderGraph::default());
        }
        let options = Options {
            seed: talechime_core::SeedPolicy::Pinned(42).resolve(0, 0),
            max_frames: 187,
            ..Default::default()
        };
        let mut pcm_bits = Vec::new();
        let mut patch_bits = Vec::new();
        let mut chunk_lengths = Vec::new();
        model.timings = Timings::default();
        let result = model.generate_inner(
            GenerateRequest {
                text: text.trim(),
                reference: None,

                options: &options,
            },
            &|| false,
            &mut |pcm| {
                if pcm.iter().any(|v| !v.is_finite()) {
                    candle_core::bail!("nonfinite graph PCM");
                }
                pcm_bits.extend(pcm.iter().map(|v| v.to_bits()));
                chunk_lengths.push(pcm.len());
                Ok(())
            },
            &mut |patch| {
                patch_bits.push(
                    patch
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?
                        .into_iter()
                        .map(f32::to_bits)
                        .collect::<Vec<_>>(),
                );
                Ok(())
            },
        );
        // Match the public request's cleanup even when generation fails.
        model.codec.reset();
        anyhow::ensure!(result? == Outcome::Eos && !pcm_bits.is_empty());
        let graph = model.cuda_graph_status();
        anyhow::ensure!(
            graph.cfm && graph.decoder == graph_enabled,
            "graph fell back"
        );
        let signature = (pcm_bits, patch_bits, chunk_lengths);
        if let Some(expected) = &baseline {
            anyhow::ensure!(
                &signature == expected,
                "decoder graph changed latent/PCM/chunks"
            );
        }
        rows.push(serde_json::json!({
            "decoder_graph":graph_enabled,"cfm_graph":graph.cfm,
            "latent_patches":signature.1.len(),"raw_pcm_samples":signature.0.len(),
            "chunk_lengths":signature.2,"natural_eos":true,"bits_equal_eager":true
        }));
        if baseline.is_none() {
            baseline = Some(signature);
        }
    }
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output)?;
    file.write_all(
        serde_json::to_string_pretty(&serde_json::json!({
            "kind":"model_inner_correctness_not_production_timing",
            "source_text_sha256":talechime_protocol::text_hash(&text),"rows":rows
        }))?
        .as_bytes(),
    )?;
    Ok(())
}
