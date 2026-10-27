//! Mandarin 1.1 frontend using exported official dictionaries and rules.
mod normalize;
mod sandhi;
mod segment;
pub use normalize::decimal_digit;

use candle_core::Result;
use jieba_rs::Jieba;
use serde::Deserialize;
use std::{
    collections::{HashMap, HashSet},
    io::BufReader,
    ops::Range,
    path::Path,
};

#[derive(Deserialize)]
struct Data {
    format: String,
    source_revision: String,
    syllables: Vec<[String; 2]>,
    words: HashMap<String, Vec<usize>>,
    phoneme_map: HashMap<String, String>,
    neutral_words: HashSet<String>,
    numeric_chars: HashSet<char>,
    not_neutral_words: HashSet<String>,
    must_erhua: HashSet<String>,
    not_erhua: HashSet<String>,
    pos_hmm: segment::PosHmm,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frontend() -> Frontend {
        let data:Data=serde_json::from_value(serde_json::json!({
            "format":"kokoro-misaki-zh-v1","source_revision":"test-only",
            "syllables":[["n","i3"],["h","ao3"],["t","ian1"],["q","i4"],["h","en3"]],
            "words":{"你":[0],"好":[1],"天":[2],"气":[3],"很":[4]},
            "phoneme_map":{"n":"ㄋ","i":"ㄧ","h":"ㄏ","ao":"ㄠ","t":"ㄊ","ian":"言","q":"ㄑ","en":"ㄣ"},
            "neutral_words":[],"not_neutral_words":[],"numeric_chars":[],"must_erhua":[],"not_erhua":[],
            "pos_hmm":{"states":[["S","n"]],"start":[0.0],"transitions":[{"0":0.0}],"emissions":[{}],"char_states":{}}
        })).unwrap();
        let mut jieba = Jieba::empty();
        let mut tags = HashMap::new();
        for word in ["你", "好", "天", "气", "很"] {
            jieba.add_word(word, Some(10), Some("n"));
            tags.insert(word.into(), (10, "n".into()));
        }
        Frontend {
            data,
            jieba,
            tags,
            max_word_chars: 1,
        }
    }

    #[test]
    fn bounded_segments_cover_original_utf8_without_truncation() -> Result<()> {
        let frontend = frontend();
        let text = format!("\n{}  ", "你好！\n天气很好。".repeat(80));
        let segments = frontend.segments(&text, 20)?;
        assert!(segments.len() > 2);
        let mut cursor = 0;
        let mut restored = String::new();
        for segment in &segments {
            assert_eq!(segment.range.start, cursor);
            assert!(segment.phonemes.chars().count() <= 20 && !segment.phonemes.is_empty());
            restored.push_str(&text[segment.range.clone()]);
            cursor = segment.range.end;
        }
        assert_eq!(cursor, text.len());
        assert_eq!(restored, text);
        Ok(())
    }

    #[test]
    fn unsupported_input_and_invalid_budgets_are_explicit() -> Result<()> {
        let frontend = frontend();
        assert!(frontend.phonemes("你好Kokoro").is_err());
        assert!(frontend.phonemes("你好🙂").is_err());
        assert!(frontend.segments("你好", 0).is_err());
        assert!(frontend.segments("你好", 511).is_err());
        assert!(frontend.segments("2026", 3).is_err());
        let error = frontend
            .segments("２０２６", 3)
            .err()
            .expect("cannot split a full-width number");
        assert!(
            error
                .to_string()
                .contains("without breaking a numeric expression")
        );
        assert!(frontend.segments(" \n\r\n ", 20)?.is_empty());
        Ok(())
    }

    #[test]
    fn e_erhua_consumes_the_marker_when_rendered_as_er() -> Result<()> {
        let mut frontend = frontend();
        let initial = frontend.data.syllables.len();
        frontend
            .data
            .syllables
            .extend([["zh".into(), "e4".into()], ["".into(), "er2".into()]]);
        frontend
            .data
            .words
            .insert("这儿".into(), vec![initial, initial + 1]);
        frontend
            .data
            .phoneme_map
            .extend([("zh".into(), "ㄓ".into()), ("er".into(), "ㄦ".into())]);
        frontend.max_word_chars = 2;
        frontend.jieba.add_word("这儿", Some(100), Some("r"));
        frontend.tags.insert("这儿".into(), (100, "r".into()));
        assert_eq!(frontend.phonemes("这儿")?, "ㄓㄦ4");
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct Word {
    text: String,
    tag: String,
}

/// A complete spoken segment bound to an original UTF-8 source range.
pub struct Segment {
    /// Byte range in the original immutable input.
    pub range: Range<usize>,
    /// Official v1.1 phoneme sequence for this segment.
    pub phonemes: String,
}

/// Official Mandarin dictionaries, segmentation and rule-based pronunciation.
pub struct Frontend {
    data: Data,
    jieba: Jieba,
    max_word_chars: usize,
    tags: HashMap<String, (u64, String)>,
}

impl Frontend {
    /// Split complete original text by phoneme budget, preserving original byte
    /// ranges. No normalized or phoneme offset is substituted for a source byte.
    pub fn segments(&self, text: &str, max_phonemes: usize) -> Result<Vec<Segment>> {
        if !(1..=510).contains(&max_phonemes) {
            candle_core::bail!("Kokoro phoneme budget must be 1..=510");
        }
        if text.len() > 16 * 1024 * 1024 {
            candle_core::bail!("Kokoro source exceeds frontend input budget");
        }
        let mut output: Vec<Segment> = Vec::new();
        let mut cursor = 0;
        let mut pending_start = 0;
        while cursor < text.len() {
            let mut candidates: Vec<_> = text[cursor..]
                .char_indices()
                .skip(1)
                .take(max_phonemes)
                .map(|(i, _)| cursor + i)
                .collect();
            if candidates.len() < max_phonemes {
                candidates.push(text.len());
            }
            let boundaries: Vec<_> = candidates
                .into_iter()
                .filter(|&i| {
                    // Do not change numeric reading by cutting a decimal/date digit run.
                    let before = text[..i].chars().next_back();
                    let after = text[i..].chars().next();
                    !before.zip(after).is_some_and(|(a, b)| {
                        decimal_digit(a).is_some()
                            && (decimal_digit(b).is_some() || ".%/".contains(b))
                            || decimal_digit(b).is_some() && ".-/".contains(a)
                    })
                })
                .collect();
            if boundaries.is_empty() {
                candle_core::bail!(
                    "Kokoro cannot split input without breaking a numeric expression"
                );
            }
            let mut lower = 0;
            let mut upper = boundaries.len();
            let mut best = None;
            while lower < upper {
                let middle = (lower + upper) / 2;
                let end = boundaries[middle];
                let phonemes = self.phonemes(&text[cursor..end])?;
                if phonemes.chars().count() <= max_phonemes {
                    best = Some((end, phonemes));
                    lower = middle + 1;
                } else {
                    upper = middle;
                }
            }
            let Some((mut end, mut phonemes)) = best else {
                candle_core::bail!("one source unit exceeds the Kokoro phoneme budget");
            };
            // Prefer a completed sentence when the budget would otherwise cut it.
            if end < text.len()
                && let Some((position, c)) = text[cursor..end]
                    .char_indices()
                    .rev()
                    .find(|(_, c)| "。！？；!?;\n".contains(*c))
            {
                let preferred = cursor + position + c.len_utf8();
                let candidate = self.phonemes(&text[cursor..preferred])?;
                if !candidate.is_empty() && candidate.chars().count() <= max_phonemes {
                    end = preferred;
                    phonemes = candidate;
                }
            }
            if phonemes.is_empty() {
                if let Some(previous) = output.last_mut() {
                    previous.range.end = end;
                    pending_start = end;
                }
            } else {
                output.push(Segment {
                    range: pending_start..end,
                    phonemes,
                });
                pending_start = end;
            }
            cursor = end;
        }
        Ok(output)
    }

    /// Load the output of `scripts/export_frontend.py`; no download or Python.
    pub fn load(root: &Path) -> Result<Self> {
        let data: Data = serde_json::from_slice(
            &std::fs::read(root.join("frontend.json")).map_err(candle_core::Error::wrap)?,
        )
        .map_err(candle_core::Error::wrap)?;
        if data.format != "kokoro-misaki-zh-v1"
            || data.source_revision != "fba1236595f2d2bf21d414ba6e57d25256afada3"
        {
            candle_core::bail!("unsupported Kokoro frontend resource format or source revision");
        }
        if data.words.is_empty()
            || data
                .words
                .values()
                .flatten()
                .any(|&i| i >= data.syllables.len())
        {
            candle_core::bail!("invalid Kokoro pronunciation dictionary");
        }
        let dictionary =
            std::fs::read(root.join("jieba.dict")).map_err(candle_core::Error::wrap)?;
        let mut jieba = Jieba::empty();
        jieba
            .load_dict(&mut BufReader::new(dictionary.as_slice()))
            .map_err(candle_core::Error::wrap)?;
        let dictionary = std::str::from_utf8(&dictionary).map_err(candle_core::Error::wrap)?;
        let mut tags = HashMap::new();
        for line in dictionary.lines() {
            let mut fields = line.split_whitespace();
            if let (Some(word), Some(freq), Some(tag)) =
                (fields.next(), fields.next(), fields.next())
            {
                tags.insert(
                    word.to_owned(),
                    (
                        freq.parse::<u64>().map_err(candle_core::Error::wrap)?,
                        tag.to_owned(),
                    ),
                );
            }
        }
        let max_word_chars = data
            .words
            .keys()
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(1);
        Ok(Self {
            data,
            jieba,
            max_word_chars,
            tags,
        })
    }

    /// Normalize and pronounce one complete Mandarin segment. Unsupported input
    /// fails explicitly instead of being filtered out of the model vocabulary.
    pub fn phonemes(&self, text: &str) -> Result<String> {
        let normalized = normalize::normalize(text)?;
        self.phonemes_normalized(&normalized, None)
    }

    /// Apply explicit g2pW polyphone predictions before the official sandhi and
    /// v1.1 mapping. Baseline [`Self::phonemes`] never invokes the context model.
    #[cfg(feature = "g2pw")]
    pub fn phonemes_with_context(
        &self,
        text: &str,
        context: &mut crate::context::ContextModel,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<String> {
        let normalized = normalize::normalize(text)?;
        if normalized.chars().any(|c| c.is_ascii_alphabetic()) {
            candle_core::bail!("Kokoro Mandarin frontend does not yet support embedded English");
        }
        let mut readings = vec![None; normalized.chars().count()];
        for prediction in context.predict(&normalized, cancelled)? {
            let target = readings.get_mut(prediction.index).ok_or_else(|| {
                candle_core::Error::Msg("context prediction lost source-character mapping".into())
            })?;
            *target = Some(prediction.reading);
        }
        self.phonemes_normalized(&normalized, Some(&readings))
    }

    fn phonemes_normalized(
        &self,
        normalized: &str,
        predictions: Option<&[Option<[String; 2]>]>,
    ) -> Result<String> {
        if normalized.is_empty() {
            return Ok(String::new());
        }
        if normalized.chars().any(|c| c.is_ascii_alphabetic()) {
            candle_core::bail!("Kokoro Mandarin frontend does not yet support embedded English");
        }
        let words = self.pos_segment(normalized)?;
        let words = self.merge(words)?;
        if predictions.is_some_and(|p| p.len() != normalized.chars().count())
            || predictions.is_some()
                && words.iter().map(|w| w.text.as_str()).collect::<String>() != normalized
        {
            candle_core::bail!("context pronunciation lost normalized character mapping");
        }
        let mut tokens: Vec<(String, String, String)> = Vec::new();
        let mut position = 0;
        for mut word in words {
            let word_start = position;
            position += word.text.chars().count();
            let punctuation = ";:,.!?—…\"()“”";
            if word.tag == "x"
                && word
                    .text
                    .chars()
                    .all(|c| ('\u{4e00}'..='\u{9fff}').contains(&c))
            {
                word.tag = "X".into();
            } else if word.tag != "x" && punctuation.contains(word.text.as_str()) {
                word.tag = "x".into();
            }
            if word.tag == "x" || word.tag == "eng" {
                if word.text.chars().all(char::is_whitespace) {
                    if let Some(last) = tokens.last_mut() {
                        last.2.push_str(&word.text);
                    }
                } else if punctuation.contains(word.text.as_str()) {
                    tokens.push((word.tag, word.text, String::new()));
                } else {
                    candle_core::bail!("unsupported Kokoro Mandarin symbol {:?}", word.text);
                }
                continue;
            }
            if let Some(last) = tokens.last_mut()
                && last.0 != "x"
                && last.0 != "eng"
                && last.2.is_empty()
            {
                last.2.push('/');
            }
            let mut phones = self.readings(&word.text)?;
            if let Some(predictions) = predictions {
                for (index, phone) in phones.iter_mut().enumerate() {
                    if let Some(Some(reading)) = predictions.get(word_start + index) {
                        *phone = reading.clone();
                    }
                }
                for (character, phone) in word.text.chars().zip(&mut phones) {
                    if character == '嗯' {
                        phone[1] = "n2".into();
                    }
                }
            }
            self.modify_tones(&word, &mut phones)?;
            self.erhua(&word, &mut phones);
            let mut phonemes = String::new();
            for [initial, mut final_] in phones {
                if final_.starts_with('i')
                    && final_
                        .get(1..)
                        .is_some_and(|s| s.chars().all(|c| c.is_ascii_digit()))
                {
                    if ["z", "c", "s"].contains(&initial.as_str()) {
                        final_ = format!("ii{}", &final_[1..]);
                    } else if ["zh", "ch", "sh", "r"].contains(&initial.as_str()) {
                        final_ = format!("iii{}", &final_[1..]);
                    }
                }
                if !initial.is_empty() {
                    phonemes.push_str(self.map(&initial)?);
                }
                let tone = final_
                    .pop()
                    .ok_or_else(|| candle_core::Error::Msg("empty Mandarin final".into()))?;
                let mut erhua = final_.ends_with('R');
                if erhua {
                    final_.pop();
                }
                if final_ == "e" && erhua {
                    final_ = "er".into();
                    erhua = false;
                }
                phonemes.push_str(self.map(&final_)?);
                if erhua {
                    phonemes.push('R');
                }
                phonemes.push(tone);
            }
            tokens.push((word.tag, phonemes, String::new()));
        }
        Ok(tokens.into_iter().map(|(_, p, w)| p + &w).collect())
    }

    fn map(&self, p: &str) -> Result<&str> {
        self.data
            .phoneme_map
            .get(p)
            .map(String::as_str)
            .ok_or_else(|| {
                candle_core::Error::Msg(format!("unsupported Mandarin phoneme component {p:?}"))
            })
    }

    fn readings(&self, word: &str) -> Result<Vec<[String; 2]>> {
        let positions: Vec<_> = word
            .char_indices()
            .map(|(i, _)| i)
            .chain(std::iter::once(word.len()))
            .collect();
        let mut cursor = 0;
        let mut result = Vec::new();
        while cursor + 1 < positions.len() {
            let end = (cursor + self.max_word_chars).min(positions.len() - 1);
            let found = (cursor + 1..=end).rev().find_map(|end| {
                self.data
                    .words
                    .get(&word[positions[cursor]..positions[end]])
                    .map(|ids| (end, ids))
            });
            let Some((end, ids)) = found else {
                candle_core::bail!(
                    "no official Mandarin pronunciation for {:?}",
                    &word[positions[cursor]..positions[cursor + 1]]
                );
            };
            result.extend(ids.iter().map(|&i| self.data.syllables[i].clone()));
            cursor = end;
        }
        Ok(result)
    }

    fn split_word(&self, word: &str) -> (String, String) {
        let mut words = self.jieba.cut_for_search(word, true);
        words.sort_by_key(|token| token.word.chars().count());
        let Some(first) = words.first() else {
            return (word.into(), String::new());
        };
        if let Some(rest) = word.strip_prefix(first.word) {
            (first.word.into(), rest.into())
        } else {
            let end = word.len() - first.word.len();
            (word[..end].into(), first.word.into())
        }
    }

    fn erhua(&self, word: &Word, phones: &mut Vec<[String; 2]>) {
        if !word.text.ends_with('儿') || phones.is_empty() {
            return;
        }
        let end = phones.len() - 1;
        if phones[end][1] == "er1" {
            phones[end][1] = "er2".into();
        }
        if !self.data.must_erhua.contains(&word.text)
            && (self.data.not_erhua.contains(&word.text)
                || ["a", "j", "nr"].contains(&word.tag.as_str()))
        {
            return;
        }
        let tail: String = word
            .text
            .chars()
            .rev()
            .take(2)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if end > 0
            && ["er2", "er5"].contains(&phones[end][1].as_str())
            && !self.data.not_erhua.contains(&tail)
        {
            phones.pop();
            let previous = &mut phones[end - 1][1];
            let tone = previous.pop().unwrap_or('5');
            previous.push('R');
            previous.push(tone);
        }
    }
}
