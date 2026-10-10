//! Exact upstream user message template and reference audio placeholders.
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
    let mut row = vec![config.audio_pad_code; config.n_vq + 1];
    row[0] = config.audio_start_token_id;
    rows.push(row);
    Ok(rows)
}

/// Official continuation truncates immediately after the assistant audio prefix:
/// there is no audio-end or message-end row until generation completes.
pub(crate) fn append_continuation(
    rows: &mut Vec<Vec<u32>>,
    config: &SpeechConfig,
    codes: &[Vec<u32>],
) -> anyhow::Result<()> {
    anyhow::ensure!(!codes.is_empty(), "empty MOSS continuation codes");
    for codes in codes {
        anyhow::ensure!(
            codes.len() == config.n_vq && codes.iter().all(|id| *id < config.audio_pad_code),
            "incompatible MOSS continuation codes"
        );
        let mut row = vec![config.audio_assistant_gen_slot_token_id];
        row.extend_from_slice(codes);
        rows.push(row);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn local_assistant_prefix_has_no_end_or_user_audio_slots() -> anyhow::Result<()> {
        let transformer: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/upstream.json"))?;
        let config: SpeechConfig = serde_json::from_value(serde_json::json!({
            "language_config": transformer["config"], "n_vq": 32,
            "audio_vocab_size": 1024, "audio_pad_code": 1024,
            "audio_start_token_id": 10, "audio_end_token_id": 11,
            "audio_user_slot_token_id": 12, "audio_assistant_gen_slot_token_id": 13,
            "sampling_rate": 24000
        }))?;
        let mut start = vec![1024; 33];
        start[0] = 10;
        let mut rows = vec![start.clone()];
        let codes = vec![vec![42; 32], vec![99; 32]];
        append_continuation(&mut rows, &config, &codes)?;
        assert_eq!(rows[0], start);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[1][0], 13);
        assert_eq!(rows[2][0], 13);
        assert_eq!(rows[1][1..], codes[0]);
        assert_eq!(rows[2][1..], codes[1]);
        assert!(append_continuation(&mut rows, &config, &[]).is_err());
        assert!(append_continuation(&mut rows, &config, &[vec![1; 16]]).is_err());
        assert!(append_continuation(&mut rows, &config, &[vec![1024; 32]]).is_err());
        Ok(())
    }
}
