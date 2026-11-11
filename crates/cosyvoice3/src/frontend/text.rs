//! Fixed CosyVoice frontend's public WeText path, without ttsfrd/Python.
//! Alibaba Inc (Xiang Lyu, Zhihao Du), Apache-2.0, revision 074ca6dc.
mod numbers;
use candle_core::Result;
use std::{path::Path, sync::OnceLock};
use talechime_text::wetext::WeText;

/// Model-owned normalization and official paragraph grouping.
pub struct TextFrontend {
    normalizer: WeText,
}
impl TextFrontend {
    /// Load the verified WeText 0.1.0 resource directory.
    pub fn load(directory: &Path) -> Result<Self> {
        Ok(Self {
            normalizer: WeText::load(directory).map_err(text_error)?,
        })
    }
    /// Keep model control tokens verbatim, otherwise normalize and group using
    /// the fixed 80/60/20 token policy. The caller supplies the actual tokenizer.
    pub fn prepare(
        &self,
        text: &str,
        enabled: bool,
        tokenize: impl Fn(&str) -> Result<usize>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<String>> {
        if cancelled() {
            candle_core::bail!("CosyVoice text processing cancelled");
        }
        if !enabled || text.is_empty() || (text.contains("<|") && text.contains("|>")) {
            return Ok(vec![text.to_owned()]);
        }
        let chinese = text
            .chars()
            .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch));
        let mut text = self
            .normalizer
            .normalize_with_cancel(text, cancelled)
            .map_err(text_error)?;
        if chinese {
            text = text.replace('\n', "");
            let chars: Vec<_> = text.chars().collect();
            text = chars
                .iter()
                .enumerate()
                .filter_map(|(index, &ch)| {
                    if ch != ' '
                        || index > 0
                            && index + 1 < chars.len()
                            && chars[index - 1].is_ascii()
                            && chars[index - 1] != ' '
                            && chars[index + 1].is_ascii()
                            && chars[index + 1] != ' '
                    {
                        Some(ch)
                    } else {
                        None
                    }
                })
                .collect();
            text = text
                .replace('²', "平方")
                .replace('³', "立方")
                .replace('.', "。")
                .replace(" - ", "，")
                .replace(['（', '）', '【', '】', '`'], "")
                .replace("——", " ");
            static COMMA: OnceLock<regex::Regex> = OnceLock::new();
            text = COMMA
                .get_or_init(|| regex::Regex::new(r"[，,、]+$").expect("static regex"))
                .replace_all(&text, "。")
                .into_owned();
        } else {
            text = numbers::spell_out(&text)?;
        }
        if cancelled() {
            candle_core::bail!("CosyVoice text processing cancelled");
        }
        let groups = split_paragraph(&text, chinese, tokenize)?;
        static PUNCTUATION: OnceLock<regex::Regex> = OnceLock::new();
        let punctuation = PUNCTUATION
            .get_or_init(|| regex::Regex::new(r"^[\p{P}\p{S}]*$").expect("static regex"));
        Ok(groups
            .into_iter()
            .filter(|text| !punctuation.is_match(text))
            .collect())
    }
}
fn text_error(error: talechime_text::Error) -> candle_core::Error {
    candle_core::Error::Msg(error.to_string())
}

fn split_paragraph(
    text: &str,
    chinese: bool,
    tokenize: impl Fn(&str) -> Result<usize>,
) -> Result<Vec<String>> {
    let punctuation = if chinese {
        "。？！；：、.?!;"
    } else {
        ".?!;:"
    };
    let mut chars: Vec<_> = text.chars().collect();
    if chars.is_empty() {
        return Ok(Vec::new());
    }
    if !punctuation.contains(
        *chars
            .last()
            .ok_or_else(|| candle_core::Error::Msg("empty paragraph".into()))?,
    ) {
        chars.push(if chinese { '。' } else { '.' });
    }
    let mut start = 0;
    let mut utterances: Vec<String> = Vec::new();
    for (index, &ch) in chars.iter().enumerate() {
        if !punctuation.contains(ch) {
            continue;
        }
        if start < index {
            utterances.push(chars[start..=index].iter().collect());
        }
        if chars
            .get(index + 1)
            .is_some_and(|ch| matches!(ch, '"' | '”'))
        {
            let last = utterances.last_mut().ok_or_else(|| {
                candle_core::Error::Msg("CosyVoice punctuation has no preceding text".into())
            })?;
            last.push(chars[index + 1]);
            start = index + 2;
        } else {
            start = index + 1;
        }
    }
    let length = |text: &str| {
        if chinese {
            Ok(text.chars().count())
        } else {
            tokenize(text)
        }
    };
    let mut groups: Vec<String> = Vec::new();
    let mut current = String::new();
    for utterance in utterances {
        if length(&format!("{current}{utterance}"))? > 80 && length(&current)? > 60 {
            groups.push(std::mem::take(&mut current));
        }
        current.push_str(&utterance);
    }
    if !current.is_empty() {
        if length(&current)? < 20 && !groups.is_empty() {
            groups
                .last_mut()
                .ok_or_else(|| candle_core::Error::Msg("missing paragraph group".into()))?
                .push_str(&current);
        } else {
            groups.push(current);
        }
    }
    Ok(groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn grouping_uses_official_limits_and_preserves_quotes() {
        let text = format!("{}。{}。短句。", "甲".repeat(65), "乙".repeat(65));
        let groups = split_paragraph(&text, true, |_| unreachable!()).unwrap();
        assert_eq!(groups.len(), 2);
        assert_eq!(groups.concat(), text);
        assert_eq!(
            split_paragraph("他说：\"你好。”下一句", true, |_| unreachable!())
                .unwrap()
                .concat(),
            "他说：\"你好。”下一句。"
        );
    }

    #[test]
    #[ignore = "requires explicit pinned WeText resources and actual CosyVoice tokenizer"]
    fn complete_public_corpus_matches_the_independent_official_frontend() {
        let tn = std::env::var_os("TALECHIME_COSY_WETEXT").expect("fixed WeText directory");
        let tokenizer =
            std::env::var_os("TALECHIME_COSY_TOKENIZER").expect("fixed model tokenizer");
        let tokenizer = tokenizers::Tokenizer::from_file(Path::new(&tokenizer)).unwrap();
        let frontend = TextFrontend::load(Path::new(&tn)).unwrap();
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/text-frontend-oracle.json"
        ))
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            let actual = frontend.prepare(
                text,
                true,
                |text| {
                    Ok(tokenizer
                        .encode(text, false)
                        .map_err(|error| candle_core::Error::Msg(error.to_string()))?
                        .len())
                },
                &|| false,
            );
            if case.get("error").is_some() {
                assert!(actual.is_err(), "{text:?}");
            } else {
                assert_eq!(
                    serde_json::to_value(actual.unwrap()).unwrap(),
                    case["output"],
                    "{text:?}"
                );
            }
        }
    }
}
