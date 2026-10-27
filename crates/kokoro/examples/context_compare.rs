//! Compare explicit independent official g2pW fixtures, including all ONNX inputs.
#[cfg(feature = "g2pw")]
fn main() -> candle_core::Result<()> {
    use candle_core::{DType, Device};
    use std::path::PathBuf;
    use talechime_kokoro::{Trace, context::ContextModel, frontend::Frontend};
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .expect("context_compare CONTEXT_EXPORT [FRONTEND_EXPORT]");
    let reference: serde_json::Value = serde_json::from_slice(
        &std::fs::read(root.join("reference.json")).map_err(candle_core::Error::wrap)?,
    )
    .map_err(candle_core::Error::wrap)?;
    let tensors =
        candle_core::safetensors::load(root.join("official-context.safetensors"), &Device::Cpu)?;
    let mut model = ContextModel::load(&root)?;
    let frontend = std::env::args_os()
        .nth(2)
        .map(PathBuf::from)
        .map(|p| Frontend::load(&p))
        .transpose()?;
    let mut correct = 0;
    let mut changes = 0;
    let cases = reference["cases"].as_array().expect("cases");
    for (number, case) in cases.iter().enumerate() {
        let text = case["normalized"].as_str().expect("normalized");
        let mut trace = Trace::enabled();
        let predictions = model.predict_traced(text, &|| false, &mut trace)?;
        let expected = case["predictions"].as_array().expect("predictions");
        for prediction in &predictions {
            assert_eq!(
                Some(prediction.pinyin.as_str()),
                expected[prediction.index].as_str(),
                "case {number}, position {}",
                prediction.index
            );
        }
        let traced = trace.tensors().expect("enabled");
        assert_eq!(traced.len(), case["batches"].as_u64().unwrap() as usize * 7);
        for (name, actual) in traced {
            let key = format!("case.{number}.{name}");
            let official = tensors.get(&key).expect(&key);
            assert_eq!(actual.dims(), official.dims(), "{key}");
            let actual = actual
                .to_dtype(DType::F64)?
                .flatten_all()?
                .to_vec1::<f64>()?;
            let official = official
                .to_dtype(DType::F64)?
                .flatten_all()?
                .to_vec1::<f64>()?;
            let tolerance = if name.ends_with("probabilities") {
                1e-5
            } else {
                0.0
            };
            let max = actual
                .iter()
                .zip(&official)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(max <= tolerance, "{key}: max difference {max}");
        }
        let index = case["index"].as_u64().unwrap() as usize;
        let actual = predictions
            .iter()
            .find(|p| p.index == index)
            .expect("target prediction");
        let gold = case["gold"].as_str().unwrap();
        correct += usize::from(actual.pinyin == gold);
        println!(
            "PASS official case {number}: {text} target={} gold={gold}",
            actual.pinyin
        );
        if let Some(frontend) = &frontend {
            let baseline = frontend.phonemes(text)?;
            let contextual = frontend.phonemes_with_context(text, &mut model, &|| false)?;
            if baseline != contextual {
                changes += 1;
                println!("  pronunciation changed: {baseline} -> {contextual}");
            }
        }
    }
    println!(
        "official input/output parity: {}/{}; manual target gold: {correct}/{}; frontend changes: {changes}",
        cases.len(),
        cases.len(),
        cases.len()
    );
    Ok(())
}

#[cfg(not(feature = "g2pw"))]
fn main() {
    eprintln!("context_compare requires --features g2pw");
    std::process::exit(2);
}
