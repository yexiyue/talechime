//! Official Nano voice-clone rows with a fixed user reference.
use serde_json::Value;
pub(super) fn rows(
    manifest: &Value,
    tokens: &[i32],
    codes: &[Vec<i32>],
) -> anyhow::Result<Vec<i32>> {
    let config = &manifest["tts_config"];
    let templates = &manifest["prompt_templates"];
    let mut rows = Vec::new();
    let text_row = |rows: &mut Vec<i32>, token: i32| {
        rows.push(token);
        rows.extend([1024; 16]);
    };
    let append = |rows: &mut Vec<i32>, key: &str| -> anyhow::Result<()> {
        for token in templates[key]
            .as_array()
            .ok_or_else(|| anyhow::anyhow!("invalid prompt template"))?
        {
            text_row(
                rows,
                token
                    .as_i64()
                    .ok_or_else(|| anyhow::anyhow!("invalid token"))? as i32,
            );
        }
        Ok(())
    };
    // The manifest stores encoded template text only. The official builder
    // separately prepends the user message's im_start control token.
    text_row(
        &mut rows,
        config["im_start_token_id"]
            .as_i64()
            .ok_or_else(|| anyhow::anyhow!("missing Nano im_start token"))? as i32,
    );
    append(&mut rows, "user_prompt_prefix_token_ids")?;
    {
        text_row(
            &mut rows,
            config["audio_start_token_id"].as_i64().unwrap_or(6) as i32,
        );
        for code in codes {
            anyhow::ensure!(code.len() == 16, "invalid voice code width");
            rows.push(8);
            rows.extend(code);
        }
        text_row(&mut rows, 7);
    }
    append(&mut rows, "user_prompt_after_reference_token_ids")?;
    for &token in tokens {
        text_row(&mut rows, token);
    }
    append(&mut rows, "assistant_prompt_prefix_token_ids")?;
    text_row(&mut rows, 6);
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires pinned Nano sentencepiece tokenizer"]
    fn public_raw_corpus_matches_official_voice_clone_rows() -> anyhow::Result<()> {
        use sha2::{Digest, Sha256};
        let path = std::env::var("TALECHIME_NANO_TOKENIZER")?;
        let bytes = std::fs::read(&path)?;
        let fixture: Value = serde_json::from_str(include_str!("assets/nano-corpus-prompt.json"))?;
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            fixture["sources_sha256"]["tokenizer"]
        );
        let tokenizer = sentencepiece_rs::SentencePieceProcessor::open(path)?;
        let manifest: Value = serde_json::from_str(include_str!("assets/nano-prompt.json"))?;
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 19);
        for case in cases {
            let text = case["source_text"].as_str().unwrap();
            assert_eq!(
                format!("{:x}", Sha256::digest(text.as_bytes())),
                case["source_sha256"]
            );
            let tokens: Vec<i32> = tokenizer
                .encode_to_ids(text)?
                .into_iter()
                .map(|id| id as i32)
                .collect();
            assert_eq!(
                serde_json::to_value(&tokens)?,
                case["text_ids"],
                "{}",
                case["id"]
            );
            let codes: Vec<Vec<i32>> = serde_json::from_value(case["reference_codes"].clone())?;
            let expected: Vec<Vec<i32>> = serde_json::from_value(case["rows"].clone())?;
            assert_eq!(
                rows(&manifest, &tokens, &codes)?,
                expected.into_iter().flatten().collect::<Vec<_>>(),
                "{}",
                case["id"]
            );
        }
        Ok(())
    }
    #[test]
    fn fixed_voice_clone_matches_pinned_official_builder() {
        let manifest: Value =
            serde_json::from_str(include_str!("assets/nano-prompt.json")).unwrap();
        let reference: Value =
            serde_json::from_str(include_str!("assets/nano-prompt-reference.json")).unwrap();
        for case in reference["cases"].as_array().unwrap() {
            let tokens: Vec<i32> = serde_json::from_value(case["tokens"].clone()).unwrap();
            let codes: Vec<Vec<i32>> = serde_json::from_value(case["codes"].clone()).unwrap();
            let expected: Vec<Vec<i32>> = serde_json::from_value(case["rows"].clone()).unwrap();
            assert_eq!(
                rows(&manifest, &tokens, &codes).unwrap(),
                expected.into_iter().flatten().collect::<Vec<_>>()
            );
        }
    }
}
