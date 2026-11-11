//! Exact upstream user message template and reference audio placeholders.
mod normalize;
use crate::config::SpeechConfig;
use tokenizers::Tokenizer;

pub fn text_prompt(
    tokenizer: &Tokenizer,
    config: &SpeechConfig,
    text: &str,
    instruction: Option<&str>,
    reference: Option<&[Vec<u32>]>,
) -> anyhow::Result<Vec<Vec<u32>>> {
    anyhow::ensure!(!text.trim().is_empty(), "empty MOSS input");
    let text = normalize::text(text)?;
    anyhow::ensure!(!text.is_empty(), "empty normalized MOSS input");
    let reference_marker = if reference.is_some() {
        "[S1]:\n<|audio|>"
    } else {
        "None"
    };
    let user = format!(
        "<user_inst>\n- Reference(s):\n{reference_marker}\n- Instruction:\n{}\n- Tokens:\nNone\n- Quality:\nNone\n- Sound Event:\nNone\n- Ambient Sound:\nNone\n- Language:\nNone\n- Text:\n{text}\n</user_inst>",
        instruction.unwrap_or("None")
    );
    let user = format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n");
    let mut rows = Vec::new();
    let mut encode = |text: &str| -> anyhow::Result<()> {
        for &token in tokenizer
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
        {
            let mut row = vec![config.audio_pad_code; config.n_vq + 1];
            row[0] = token;
            rows.push(row);
        }
        Ok(())
    };
    if let Some(reference) = reference {
        let (before, after) = user
            .split_once("<|audio|>")
            .expect("constructed placeholder");
        encode(before)?;
        let mut row = vec![config.audio_pad_code; config.n_vq + 1];
        row[0] = config.audio_start_token_id;
        rows.push(row);
        for frame in reference {
            anyhow::ensure!(
                frame.len() == config.n_vq && frame.iter().all(|id| *id < config.audio_pad_code),
                "incompatible MOSS reference codes"
            );
            let mut row = vec![config.audio_user_slot_token_id];
            row.extend_from_slice(frame);
            rows.push(row);
        }
        let mut row = vec![config.audio_pad_code; config.n_vq + 1];
        row[0] = config.audio_end_token_id;
        rows.push(row);
        for &token in tokenizer
            .encode(after, false)
            .map_err(|e| anyhow::anyhow!(e.to_string()))?
            .get_ids()
        {
            let mut row = vec![config.audio_pad_code; config.n_vq + 1];
            row[0] = token;
            rows.push(row);
        }
    } else {
        encode(&user)?;
    }
    if std::env::var("TALECHIME_TEXT_DIAGNOSTICS").is_ok_and(|value| value == "1") {
        eprintln!(
            "TALECHIME_TEXT_DIAGNOSTIC {}",
            serde_json::json!({"backend":"moss-design", "phase":"model_prompt",
                "execution_text":text, "tokens":rows.iter().map(|row| row[0]).collect::<Vec<_>>()})
        );
    }
    Ok(rows)
}

#[cfg(test)]
mod official_fixture_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    #[ignore = "requires pinned VoiceGenerator model tokenizer and config"]
    fn public_corpus_matches_official_with_actual_tokenizer() -> anyhow::Result<()> {
        let dir = PathBuf::from(std::env::var("MOSS_VOICEDESIGN_DIR")?);
        let config: SpeechConfig =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/voicedesign-corpus-prompt.json"
        ))?;
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 19);
        for case in cases {
            let actual = text_prompt(
                &tokenizer,
                &config,
                case["source_text"].as_str().unwrap(),
                Some(case["instruction"].as_str().unwrap()),
                None,
            )?;
            assert_eq!(
                serde_json::to_value(actual)?,
                case["rows"],
                "{}",
                case["id"]
            );
        }
        Ok(())
    }

    #[test]
    #[ignore = "requires pinned VoiceGenerator model tokenizer and config"]
    fn compare_official_no_reference_prompt() -> anyhow::Result<()> {
        let dir = PathBuf::from(std::env::var("MOSS_VOICEDESIGN_DIR")?);
        let config: SpeechConfig =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../tests/fixtures/voicedesign-official-prompt.json"
        ))?;
        let mut mismatches = 0;
        for case in fixture["cases"].as_array().unwrap() {
            let id = case["id"].as_str().unwrap();
            let expected: Vec<Vec<u32>> = serde_json::from_value(case["rows"].clone())?;
            let actual = text_prompt(
                &tokenizer,
                &config,
                case["text"].as_str().unwrap(),
                Some(case["instruction"].as_str().unwrap()),
                None,
            )?;
            let first = actual.iter().zip(&expected).position(|(a, b)| a != b);
            eprintln!(
                "{id}: official={} current={} first_difference={first:?}",
                expected.len(),
                actual.len()
            );
            mismatches += usize::from(actual != expected);
        }
        anyhow::ensure!(
            mismatches == 0,
            "{mismatches} official prompt comparisons differ"
        );
        Ok(())
    }
}
