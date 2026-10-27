//! Real-model tests are opt-in and only read explicitly supplied local resources.
use super::*;

fn fixture() -> Result<(Model, HashMap<String, Tensor>, SynthesisOptions)> {
    let root = std::env::var_os("KOKORO_OFFICIAL_FIXTURE_DIR")
        .map(std::path::PathBuf::from)
        .ok_or_else(|| {
            candle_core::Error::Msg(
                "set KOKORO_OFFICIAL_FIXTURE_DIR to an explicit local export".into(),
            )
        })?;
    let model = Model::load(
        &root.join("results/weights-folded.safetensors"),
        &root.join("config.json"),
        &Device::Cpu,
    )?;
    let reference = candle_core::safetensors::load(
        root.join("results/official-f32.safetensors"),
        &Device::Cpu,
    )?;
    let metadata: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("results/report.json")).map_err(candle_core::Error::wrap)?,
    )
    .map_err(candle_core::Error::wrap)?;
    let options = SynthesisOptions {
        speed: metadata
            .get("speed")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(1.0) as f32,
        ..Default::default()
    };
    Ok((model, reference, options))
}

fn compare_audio(actual: &Tensor, expected: &Tensor, minimum_snr: f64) -> Result<()> {
    let actual = actual.flatten_all()?.to_vec1::<f32>()?;
    let expected = expected.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(actual.len(), expected.len());
    let mut error = 0.0;
    let mut power = 0.0;
    let mut maximum = 0f32;
    for (a, b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        error += f64::from(a - b).powi(2);
        power += f64::from(b).powi(2);
        maximum = maximum.max((a - b).abs());
    }
    let snr = 10.0 * (power / error).log10();
    eprintln!(
        "samples={} max_abs={maximum:.8e} rmse={:.8e} snr_db={snr:.4}",
        actual.len(),
        (error / actual.len() as f64).sqrt()
    );
    assert!(
        error == 0.0 || snr >= minimum_snr,
        "PCM SNR {snr:.4} below {minimum_snr}"
    );
    Ok(())
}

#[test]
#[ignore = "requires an explicitly supplied official real-model export; never downloads"]
fn official_fixed_feature_decoder() -> Result<()> {
    let (model, reference, _) = fixture()?;
    let durations = reference["durations"]
        .to_dtype(DType::U32)?
        .to_vec1::<u32>()?;
    let indices: Vec<u32> = durations
        .iter()
        .enumerate()
        .flat_map(|(i, &d)| std::iter::repeat_n(i as u32, d as usize))
        .collect();
    let indices = Tensor::new(indices, &Device::Cpu)?;
    let asr = reference["text_encoder"]
        .contiguous()?
        .index_select(&indices, 2)?;
    let noise = ExcitationNoise {
        initial_phase: reference["random.0.rand"].clone(),
        gaussian: reference["random.1.randn_like"].clone(),
    };
    let audio = model.weights.decode(
        &asr,
        &reference["predictor.F0_proj"],
        &reference["predictor.N_proj"],
        &reference["style"].narrow(1, 0, 128)?,
        &noise,
        &mut Trace::default(),
    )?;
    compare_audio(&audio, &reference["audio"], 60.0)
}

#[test]
#[ignore = "requires an explicitly supplied official real-model export; never downloads"]
fn official_complete_acoustic_model() -> Result<()> {
    let (model, reference, options) = fixture()?;
    let ids = reference["input_ids"]
        .to_dtype(DType::U32)?
        .flatten_all()?
        .to_vec1::<u32>()?;
    let noise = ExcitationNoise {
        initial_phase: reference["random.0.rand"].clone(),
        gaussian: reference["random.1.randn_like"].clone(),
    };
    let actual = model.forward(
        &ids,
        &reference["style"],
        &options,
        Some(&noise),
        &mut Trace::default(),
    )?;
    assert_eq!(
        actual.durations,
        reference["durations"]
            .to_dtype(DType::U32)?
            .to_vec1::<u32>()?
    );
    compare_audio(&actual.audio, &reference["audio"], 40.0)
}
