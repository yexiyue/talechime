//! Independent Misaki/eSpeak fixtures; real frontend and optional g2pW data are external.
use candle_core::Result;
use std::path::PathBuf;
use talechime_kokoro::frontend::Frontend;

fn frontend() -> Result<Frontend> {
    let path = std::env::var_os("KOKORO_FRONTEND_DIR")
        .ok_or_else(|| candle_core::Error::Msg("set KOKORO_FRONTEND_DIR".into()))?;
    Frontend::load(&PathBuf::from(path))
}

fn fixture(json: &str) -> serde_json::Value {
    serde_json::from_str(json).expect("independent fixture")
}

#[cfg(feature = "g2pw")]
#[test]
#[ignore = "requires pinned frontend and g2pW resources"]
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
                // Historical eSpeak outputs are not the neural OOV contract.
                if id == "spanish" {
                    assert_ne!(actual, case["actual_phonemes"], "{variant} {id}");
                    continue;
                }
                if id == "long-word" {
                    assert!(
                        actual.contains(
                            &loqui_g2p::G2p::new(
                                loqui_g2p::Dialect::American,
                                loqui_g2p::OovFallback::Neural
                            )
                            .unwrap()
                            .phonemize("zorbulate")
                        )
                    );
                    continue;
                }
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
    // OOV cases intentionally use neural pronunciations. Spaced years and
    // unknown symbols are explicit product policies. The
    // strict comparison example still reports both differences.
    !matches!(
        id,
        "year_spaced_product_policy" | "unsupported" | "oov" | "oov_name"
    )
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
#[ignore = "requires pinned external Kokoro frontend export"]
fn neural_oov_works_without_external_helper() -> Result<()> {
    let frontend = frontend()?;
    let model = loqui_g2p::G2p::new(loqui_g2p::Dialect::American, loqui_g2p::OovFallback::Neural)
        .map_err(candle_core::Error::wrap)?;
    for word in ["Hermione", "zorbulate", "thraxxil"] {
        let expected = model.phonemize(word);
        assert!(!expected.is_empty());
        assert_eq!(frontend.phonemes(word)?, expected, "{word}");
    }
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
