use std::path::Path;

use anyhow::{Context, Result, bail};

pub fn device(name: &str) -> Result<candle_core::Device> {
    match name {
        "cpu" => Ok(candle_core::Device::Cpu),
        "cuda" => Ok(candle_core::Device::new_cuda(0)?),
        "metal" => Ok(candle_core::Device::new_metal(0)?),
        other => bail!("unsupported device {other}"),
    }
}

pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec)
        .with_context(|| format!("create {}", path.display()))?;
    for sample in samples {
        writer.write_sample(
            sample
                .clamp(
                    -talechime_indextts::PCM16_MAX,
                    talechime_indextts::PCM16_MAX,
                )
                .round() as i16,
        )?;
    }
    writer.finalize()?;
    Ok(())
}
