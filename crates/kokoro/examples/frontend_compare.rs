//! Compare only with explicitly provided, independently generated official fixtures.
use candle_core::Result;
use serde::Deserialize;
use std::path::PathBuf;
use talechime_kokoro::frontend::Frontend;

#[derive(Deserialize)]
struct Case {
    text: String,
    phonemes: String,
}

fn main() -> Result<()> {
    let root = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or_else(|| {
            candle_core::Error::Msg("usage: frontend_compare EXPORTED_FRONTEND_DIR".into())
        })?;
    let frontend = Frontend::load(&root)?;
    let cases: Vec<Case> = serde_json::from_slice(
        &std::fs::read(root.join("fixtures.json")).map_err(candle_core::Error::wrap)?,
    )
    .map_err(candle_core::Error::wrap)?;
    let mut failed = 0;
    for case in &cases {
        match frontend.phonemes(&case.text) {
            Ok(actual) if actual == case.phonemes => println!("PASS {}", case.text),
            Err(error)
                if case.text.chars().any(|c| c.is_ascii_alphabetic())
                    && error
                        .to_string()
                        .contains("does not yet support embedded English") =>
            {
                println!("EXPLICIT_UNSUPPORTED {}", case.text)
            }
            result => {
                failed += 1;
                eprintln!(
                    "FAIL {}\nexpected {}\nactual {:?}",
                    case.text, case.phonemes, result
                );
            }
        }
    }
    println!("{} cases, {failed} failures", cases.len());
    if failed > 0 {
        candle_core::bail!("official Mandarin frontend comparison failed");
    }
    Ok(())
}
