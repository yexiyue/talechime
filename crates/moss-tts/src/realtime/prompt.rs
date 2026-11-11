//! Fixed single-turn system/reference prompt and twelve-token lookahead.
const SYSTEM: &str = "<|im_start|>system\nYou are a highly expressive text-to-speech (TTS) engine developed by Mosi Intelligence. \nYou possess natural language understanding, emotional modeling, and multi-style speech generation capabilities, allowing you to generate the corresponding speech based on the text given in the assistant.<|im_end|>\n";

pub(super) fn initial_rows(
    mut encode: impl FnMut(&str) -> anyhow::Result<Vec<u32>>,
    text: &[u32],
    reference: Option<&[Vec<u32>]>,
) -> anyhow::Result<(Vec<Vec<u32>>, usize)> {
    anyhow::ensure!(!text.is_empty(), "empty Realtime tokenization");
    let mut prefix = SYSTEM.to_owned();
    if let Some(reference) = reference {
        anyhow::ensure!(!reference.is_empty(), "empty Realtime reference");
        prefix.push_str("<|im_start|>context\nThe assistant section should be synthesized using the following voice timbre:");
        prefix.push_str(&"<|audio_pad|>".repeat(reference.len()));
        prefix.push_str("<|im_end|>\n");
    }
    let row = |id| {
        let mut row = vec![1024; 17];
        row[0] = id;
        row
    };
    let mut rows: Vec<_> = encode(&prefix)?.into_iter().map(row).collect();
    if let Some(reference) = reference {
        let positions: Vec<_> = rows
            .iter()
            .enumerate()
            .filter_map(|(index, row)| (row[0] == 151654).then_some(index))
            .collect();
        anyhow::ensure!(
            positions.len() == reference.len(),
            "Realtime audio placeholder mismatch"
        );
        for (position, codes) in positions.into_iter().zip(reference) {
            anyhow::ensure!(
                codes.len() >= 16 && codes[..16].iter().all(|id| *id < 1024),
                "invalid Realtime reference"
            );
            rows[position][1..].copy_from_slice(&codes[..16]);
        }
    }
    rows.extend(encode("<|im_start|>assistant\n")?.into_iter().map(row));
    let consumed = text.len().min(12);
    rows.extend(text[..consumed].iter().copied().map(row));
    rows.last_mut().expect("nonempty text")[1] = 1025;
    Ok((rows, consumed))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires the fixed Realtime tokenizer"]
    fn public_corpus_matches_official_with_actual_tokenizer() -> anyhow::Result<()> {
        let path = std::env::var("MOSS_REALTIME_TOKENIZER")?;
        let tokenizer =
            tokenizers::Tokenizer::from_file(path).map_err(|e| anyhow::anyhow!(e.to_string()))?;
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/realtime-corpus-prompt.json"
        ))?;
        let cases = fixture["cases"].as_array().unwrap();
        assert_eq!(cases.len(), 38);
        for case in cases {
            let encode = |s: &str| -> anyhow::Result<Vec<u32>> {
                Ok(tokenizer
                    .encode(s, false)
                    .map_err(|e| anyhow::anyhow!(e.to_string()))?
                    .get_ids()
                    .to_vec())
            };
            let text = encode(case["source_text"].as_str().unwrap())?;
            assert_eq!(
                serde_json::to_value(&text)?,
                case["text_ids"],
                "{}",
                case["id"]
            );
            let reference: Option<Vec<Vec<u32>>> =
                serde_json::from_value(case["reference"].clone())?;
            let (rows, consumed) = initial_rows(encode, &text, reference.as_deref())?;
            assert_eq!(serde_json::to_value(rows)?, case["rows"], "{}", case["id"]);
            assert_eq!(consumed, case["consumed"].as_u64().unwrap() as usize);
        }
        Ok(())
    }

    #[test]
    fn actual_prompt_matches_fixed_official_processor() {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/realtime-initial-prompt.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let reference: Option<Vec<Vec<u32>>> =
                serde_json::from_value(case["reference"].clone()).unwrap();
            let text: Vec<u32> = serde_json::from_value(case["text_ids"].clone()).unwrap();
            let (actual, consumed) = initial_rows(
                |value| {
                    let ids = case["encodings"]
                        .get(value)
                        .expect("exact official prompt string");
                    Ok(serde_json::from_value(ids.clone()).unwrap())
                },
                &text,
                reference.as_deref(),
            )
            .unwrap();
            assert_eq!(serde_json::to_value(actual).unwrap(), case["rows"]);
            assert_eq!(consumed, case["consumed"].as_u64().unwrap() as usize);
        }
    }
}
