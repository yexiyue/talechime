//! Compare independent Misaki mixed-language fixtures without hiding mismatches.
use candle_core::Result;
use std::path::PathBuf;
use talechime_kokoro::frontend::Frontend;

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let frontend_path = PathBuf::from(args.next().expect("frontend directory"));
    let fixture_path = PathBuf::from(args.next().expect("oracle JSON path"));
    let frontend = Frontend::load(&frontend_path)?;
    let fixture: serde_json::Value =
        serde_json::from_slice(&std::fs::read(fixture_path).map_err(candle_core::Error::wrap)?)
            .map_err(candle_core::Error::wrap)?;
    #[cfg(feature = "g2pw")]
    let mut context = args
        .next()
        .map(|path| talechime_kokoro::context::ContextModel::load(&PathBuf::from(path)))
        .transpose()?;
    #[cfg(feature = "g2pw")]
    let contextual_fixture: Option<serde_json::Value> = args
        .next()
        .map(|path| {
            let bytes = std::fs::read(path).map_err(candle_core::Error::wrap)?;
            serde_json::from_slice(&bytes).map_err(candle_core::Error::wrap)
        })
        .transpose()?;
    #[cfg(feature = "g2pw")]
    if context.is_some() != contextual_fixture.is_some() {
        candle_core::bail!("provide both a g2pW directory and independent contextual fixture");
    }
    let mut failed = 0;
    for case in fixture["cases"].as_array().expect("oracle cases") {
        let id = case["id"].as_str().expect("case id");
        let text = case["text"].as_str().expect("case text");
        let expected = case["phonemes"].as_str().expect("official phonemes");
        let actual = frontend.phonemes(text);
        let ok = actual.as_deref().is_ok_and(|value| value == expected);
        println!(
            "{} baseline {id}: expected={expected:?} actual={actual:?}",
            if ok { "PASS" } else { "DIFF" }
        );
        failed += usize::from(!ok);
        #[cfg(feature = "g2pw")]
        if let Some(context) = context.as_mut() {
            let contextual_case = contextual_fixture.as_ref().unwrap()["cases"]
                .as_array()
                .and_then(|cases| {
                    cases
                        .iter()
                        .find(|item| item["id"] == id && item["text"] == text)
                })
                .ok_or_else(|| {
                    candle_core::Error::Msg(format!("missing contextual fixture {id}"))
                })?;
            let expected = contextual_case["phonemes"]
                .as_str()
                .expect("contextual phonemes");
            let actual = frontend.phonemes_with_context(text, context, &|| false);
            let ok = actual.as_deref().is_ok_and(|value| value == expected);
            println!(
                "{} g2pw {id}: expected={expected:?} actual={actual:?}",
                if ok { "PASS" } else { "DIFF" }
            );
            failed += usize::from(!ok);
        }
    }
    if failed > 0 {
        candle_core::bail!("{failed} comparisons differ from their official phoneme fixtures");
    }
    Ok(())
}
