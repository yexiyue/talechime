//! Independent Misaki/eSpeak fixtures; real frontend and optional g2pW data are external.
use candle_core::Result;
use std::path::PathBuf;
use talechime_kokoro::frontend::{EspeakConfig, Frontend};

fn frontend() -> Result<Frontend> {
    let path = std::env::var_os("KOKORO_FRONTEND_DIR")
        .ok_or_else(|| candle_core::Error::Msg("set KOKORO_FRONTEND_DIR".into()))?;
    let bundle = std::env::var_os("TALECHIME_ESPEAK_DIR")
        .ok_or_else(|| candle_core::Error::Msg("set TALECHIME_ESPEAK_DIR".into()))?;
    let options = EspeakConfig::from_bundle(&PathBuf::from(bundle))?;
    Frontend::load_with_espeak(&PathBuf::from(path), options)
}

fn fixture(json: &str) -> serde_json::Value {
    serde_json::from_str(json).expect("independent fixture")
}

#[cfg(feature = "g2pw")]
#[test]
#[ignore = "requires pinned frontend, g2pW and separately built eSpeak bundle"]
fn public_corpus_records_official_matches_and_differences() -> Result<()> {
    use sha2::{Digest, Sha256};
    let frontend = frontend()?;
    let context_path = std::env::var_os("KOKORO_CONTEXT_DIR")
        .ok_or_else(|| candle_core::Error::Msg("set KOKORO_CONTEXT_DIR".into()))?;
    let mut context = talechime_kokoro::context::ContextModel::load(&PathBuf::from(context_path))?;
    let native = fixture(include_str!("fixtures/corpus-native-comparison.json"));
    let baseline = fixture(include_str!("fixtures/corpus-misaki-espeak.json"));
    let contextual = fixture(include_str!("fixtures/corpus-g2pw-espeak.json"));
    let cases = native["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 38);
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let text = case["source_text"].as_str().unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(text.as_bytes())),
            case["source_sha256"]
        );
        let variant = case["variant"].as_str().unwrap();
        let official = if variant == "baseline" {
            &baseline
        } else {
            &contextual
        };
        let expected = official["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|item| item["id"] == id)
            .unwrap();
        assert_eq!(expected["source_text"], text);
        assert_eq!(expected["source_sha256"], case["source_sha256"]);
        let actual = if variant == "baseline" {
            frontend.phonemes(text)
        } else {
            assert_eq!(variant, "g2pw");
            frontend.phonemes_with_context(text, &mut context, &|| false)
        };
        match case["classification"].as_str().unwrap() {
            "matched" | "difference" => {
                let actual = actual?;
                assert_eq!(actual, case["actual_phonemes"], "{variant} {id}");
                assert_eq!(
                    actual == expected["phonemes"].as_str().unwrap(),
                    case["classification"] == "matched",
                    "{variant} {id}"
                );
            }
            "unsupported_rejection" => {
                let error = actual.unwrap_err().to_string();
                assert_eq!(error, case["error"], "{variant} {id}");
                assert!(error.contains("unsupported Kokoro Mandarin symbol"));
            }
            other => panic!("unknown corpus classification: {other}"),
        }
    }
    Ok(())
}

fn comparable(id: &str) -> bool {
    // Spaced years and unknown symbols are explicit product policies. The
    // strict comparison example still reports both differences.
    !matches!(id, "year_spaced_product_policy" | "unsupported")
}

#[test]
#[ignore = "requires the pinned external Kokoro frontend export"]
fn fullwidth_percent_is_an_explicit_unsupported_boundary() -> Result<()> {
    let frontend = frontend()?;
    let data = fixture(include_str!("fixtures/fullwidth-percent-misaki.json"));
    let case = &data["cases"][0];
    assert_eq!(case["phonemes"].as_str(), Some("ㄋㄧ2ㄏㄠ3❓"));
    let error = frontend
        .phonemes(case["text"].as_str().unwrap())
        .unwrap_err();
    assert!(error.to_string().contains('％'), "{error}");
    Ok(())
}

#[cfg(feature = "g2pw")]
#[test]
#[ignore = "requires the pinned external Kokoro frontend and g2pW exports"]
fn contextual_fullwidth_percent_is_unsupported() -> Result<()> {
    let frontend = frontend()?;
    let context_path = std::env::var_os("KOKORO_CONTEXT_DIR")
        .ok_or_else(|| candle_core::Error::Msg("set KOKORO_CONTEXT_DIR".into()))?;
    let mut context = talechime_kokoro::context::ContextModel::load(&PathBuf::from(context_path))?;
    let data = fixture(include_str!("fixtures/fullwidth-percent-misaki.json"));
    let error = frontend
        .phonemes_with_context(
            data["cases"][0]["text"].as_str().unwrap(),
            &mut context,
            &|| false,
        )
        .unwrap_err();
    assert!(error.to_string().contains('％'), "{error}");
    Ok(())
}

#[test]
#[ignore = "requires the pinned external Kokoro frontend export"]
fn official_mixed_reference_cases_match() -> Result<()> {
    let frontend = frontend()?;
    let data = fixture(include_str!("fixtures/mixed-misaki-espeak.json"));
    for case in data["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        if comparable(id) {
            assert_eq!(
                frontend.phonemes(case["text"].as_str().unwrap())?,
                case["phonemes"].as_str().unwrap(),
                "{id}"
            );
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires pinned frontend and separately built eSpeak CLI/data"]
fn official_espeak_oov_words_match() -> Result<()> {
    let frontend = frontend()?;
    let data = fixture(include_str!("fixtures/english-oov-espeak.json"));
    for case in data["cases"].as_array().unwrap() {
        assert_eq!(
            frontend.phonemes(case["text"].as_str().unwrap())?,
            case["phonemes"].as_str().unwrap(),
            "{}",
            case["id"]
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires pinned frontend and TALECHIME_ESPEAK_DIR bundle"]
fn default_bundle_discovery_handles_real_oov() -> Result<()> {
    let path = std::env::var_os("KOKORO_FRONTEND_DIR")
        .ok_or_else(|| candle_core::Error::Msg("set KOKORO_FRONTEND_DIR".into()))?;
    let frontend = Frontend::load(&PathBuf::from(path))?;
    let data = fixture(include_str!("fixtures/english-oov-espeak.json"));
    let case = &data["cases"][0];
    assert_eq!(
        frontend.phonemes(case["text"].as_str().unwrap())?,
        case["phonemes"].as_str().unwrap()
    );
    Ok(())
}

#[cfg(feature = "g2pw")]
#[test]
#[ignore = "requires the pinned external Kokoro frontend and g2pW exports"]
fn official_context_mixed_reference_cases_match() -> Result<()> {
    let frontend = frontend()?;
    let context_path = std::env::var_os("KOKORO_CONTEXT_DIR")
        .ok_or_else(|| candle_core::Error::Msg("set KOKORO_CONTEXT_DIR".into()))?;
    let mut context = talechime_kokoro::context::ContextModel::load(&PathBuf::from(context_path))?;
    let data = fixture(include_str!("fixtures/mixed-g2pw-espeak.json"));
    for case in data["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        if comparable(id) {
            assert_eq!(
                frontend.phonemes_with_context(
                    case["text"].as_str().unwrap(),
                    &mut context,
                    &|| false
                )?,
                case["phonemes"].as_str().unwrap(),
                "{id}"
            );
        }
    }
    Ok(())
}
