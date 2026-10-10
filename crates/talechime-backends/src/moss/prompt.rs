//! Official Nano voice-clone and assistant-prefix continuation rows.
use serde_json::Value;
pub(super) fn rows(
    manifest: &Value,
    tokens: &[i32],
    codes: &[Vec<i32>],
    continuation: Option<&[i32]>,
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
    append(&mut rows, "user_prompt_prefix_token_ids")?;
    if let Some(none_tokens) = continuation {
        for &token in none_tokens {
            text_row(&mut rows, token);
        }
    } else {
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
    if continuation.is_some() {
        for code in codes {
            anyhow::ensure!(
                code.len() == 16 && code.iter().all(|v| (0..1024).contains(v)),
                "invalid continuation audio codes"
            );
            rows.push(
                config["audio_assistant_slot_token_id"]
                    .as_i64()
                    .unwrap_or(9) as i32,
            );
            rows.extend(code);
        }
    }
    Ok(rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn official_continuation_moves_codes_to_assistant_and_has_no_user_audio() {
        let manifest: Value =
            serde_json::from_str(include_str!("assets/browser_poc_manifest.json")).unwrap();
        let codes = vec![vec![42; 16], vec![17; 16]];
        let text = [123, 456];
        let none = [789];
        let continuation = rows(&manifest, &text, &codes, Some(&none)).unwrap();
        let clone = rows(&manifest, &text, &codes, None).unwrap();
        let cr: Vec<_> = continuation.as_chunks::<17>().0.iter().collect();
        let vr: Vec<_> = clone.as_chunks::<17>().0.iter().collect();
        assert!(!cr.iter().any(|r| r[0] == 8));
        assert_eq!(cr[cr.len() - 2][0], 9);
        assert_eq!(&cr[cr.len() - 2][1..], &codes[0]);
        assert_eq!(cr.last().unwrap()[0], 9);
        let prefix = manifest["prompt_templates"]["user_prompt_prefix_token_ids"]
            .as_array()
            .unwrap()
            .len();
        assert_eq!(cr[prefix][0], 789);
        assert_eq!(vr[prefix][0], 6);
        assert_eq!(vr[prefix + 1][0], 8);
        let assistant = cr.iter().position(|r| r[0] == 123).unwrap();
        assert_eq!(cr[assistant + 1][0], 456);
        assert_eq!(cr[cr.len() - 3][0], 6);
        assert_eq!(vr.last().unwrap()[0], 6);
        assert!(
            cr.iter()
                .filter(|r| r[0] != 9)
                .all(|r| r[1..].iter().all(|v| *v == 1024))
        );
        assert!(rows(&manifest, &text, &[vec![1024; 16]], Some(&none)).is_err());
    }
}
