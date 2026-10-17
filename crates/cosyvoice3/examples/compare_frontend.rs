//! Compare official ONNX prompt features against the Python frontend.
#[cfg(feature = "ort")]
fn main() -> anyhow::Result<()> {
    use anyhow::Context;
    use candle_core::{DType, Device, Tensor};
    let args = std::env::args().collect::<Vec<_>>();
    let model = std::path::Path::new(args.get(1).context("missing converted model dir")?);
    let fixtures = std::path::Path::new(args.get(2).context("missing fixture dir")?);
    let mut wav = hound::WavReader::open(args.get(3).context("missing reference WAV")?)?;
    let spec = wav.spec();
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => wav.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => wav
            .samples::<i32>()
            .map(|s| s.map(|s| s as f32 / 2f32.powi(i32::from(spec.bits_per_sample) - 1)))
            .collect::<Result<Vec<_>, _>>()?,
    };
    let mono = samples
        .chunks_exact(spec.channels as usize)
        .map(|frame| frame.iter().sum::<f32>() / f32::from(spec.channels))
        .collect::<Vec<_>>();
    let mut frontend = cosyvoice3::frontend::ort_models::FrontendOnnx::load(model, 1)?;
    let prompt =
        cosyvoice3::pipeline::extract_prompt_features(&mut frontend, &mono, spec.sample_rate, "")?;
    let audio = Tensor::from_slice(&mono, mono.len(), &Device::Cpu)?;
    let audio16 = cosyvoice3::frontend::audio::resample(&audio, spec.sample_rate as usize, 16000)?;
    let fbank = cosyvoice3::frontend::audio::KaldiFbank::new().forward(&audio16)?;
    let embedding = Tensor::from_slice(&prompt.embedding, (1, 192), &Device::Cpu)?;
    candle_core::safetensors::save(
        &std::collections::HashMap::from([
            ("audio16", audio16),
            ("fbank", fbank),
            ("embedding", embedding),
        ]),
        fixtures.join("rust-frontend.safetensors"),
    )?;
    let expected =
        candle_core::safetensors::load(fixtures.join("frontend.safetensors"), &Device::Cpu)?;
    common::compare(
        "speech_tokens",
        &Tensor::from_slice(
            &prompt.speech_tokens,
            (1, prompt.speech_tokens.len()),
            &Device::Cpu,
        )?
        .to_dtype(DType::F32)?,
        &expected["prompt_speech_token"].to_dtype(DType::F32)?,
        0.,
    )?;
    common::compare(
        "prompt_mel",
        &Tensor::from_slice(&prompt.mel, (1, prompt.mel_frames, 80), &Device::Cpu)?,
        &expected["prompt_speech_feat"],
        3e-4,
    )?;
    common::compare(
        "speaker_embedding",
        &Tensor::from_slice(&prompt.embedding, (1, 192), &Device::Cpu)?,
        &expected["prompt_embedding"],
        3e-4,
    )?;
    Ok(())
}
#[cfg(feature = "ort")]
mod common;
#[cfg(not(feature = "ort"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!("compare_frontend requires --features ort")
}
