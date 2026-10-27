//! Adapted from pinned Misaki/PaddleSpeech tone sandhi (Apache-2.0).
//! Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
use super::{Frontend, Word};
use candle_core::Result;

fn lexical(word: &Word) -> bool {
    word.tag != "x" && word.tag != "eng"
}
fn repeated(word: &str) -> bool {
    let c: Vec<_> = word.chars().collect();
    c.len() == 2 && c[0] == c[1]
}
fn third(phones: &[[String; 2]]) -> bool {
    phones.iter().all(|p| p[1].ends_with('3'))
}
fn tone(phone: &mut [String; 2], tone: char) {
    phone[1].pop();
    phone[1].push(tone);
}
fn tail(word: &str, length: usize) -> String {
    word.chars()
        .rev()
        .take(length)
        .collect::<Vec<_>>()
        .into_iter()
        .rev()
        .collect()
}

impl Frontend {
    pub(super) fn merge(&self, words: Vec<Word>) -> Result<Vec<Word>> {
        let mut output = Vec::new();
        // 不 merges forward, preserving the original next-word POS.
        for (i, word) in words.iter().enumerate() {
            let mut word = word.clone();
            if lexical(&word) && i > 0 && words[i - 1].text == "不" {
                word.text = "不".to_owned() + &word.text;
            }
            if word.text != "不" || words.get(i + 1).is_none_or(|w| !lexical(w)) {
                output.push(word);
            }
        }
        let words = output;
        let mut output: Vec<Word> = Vec::new();
        let mut skip = false;
        for (i, word) in words.iter().enumerate() {
            if skip {
                skip = false;
                continue;
            }
            if i > 0
                && word.text == "一"
                && words.get(i + 1).is_some_and(|next| {
                    next.text == words[i - 1].text && words[i - 1].tag == "v" && lexical(next)
                })
            {
                if let Some(previous) = output.last_mut() {
                    previous.text.push('一');
                    previous.text.push_str(&words[i + 1].text);
                }
                skip = true;
            } else {
                output.push(word.clone());
            }
        }
        let words = output;
        let mut output: Vec<Word> = Vec::new();
        for word in words {
            if lexical(&word) && output.last().is_some_and(|p| p.text == "一") {
                if let Some(previous) = output.last_mut() {
                    previous.text.push_str(&word.text);
                }
            } else {
                output.push(word);
            }
        }
        let words = output;
        let mut output: Vec<Word> = Vec::new();
        for word in words {
            if lexical(&word) && output.last().is_some_and(|p| p.text == word.text) {
                if let Some(previous) = output.last_mut() {
                    previous.text.push_str(&word.text);
                }
            } else {
                output.push(word);
            }
        }
        let words = self.merge_thirds(output, false)?;
        let words = self.merge_thirds(words, true)?;
        let mut output: Vec<Word> = Vec::new();
        for word in words {
            if word.text == "儿" && output.last().is_some_and(lexical) {
                if let Some(previous) = output.last_mut() {
                    previous.text.push('儿');
                }
            } else {
                output.push(word);
            }
        }
        Ok(output)
    }

    fn merge_thirds(&self, words: Vec<Word>, boundary_only: bool) -> Result<Vec<Word>> {
        let readings = words
            .iter()
            .map(|word| {
                if lexical(word) {
                    self.readings(&word.text)
                } else {
                    Ok(vec![[String::new(), "0".into()]])
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let mut output: Vec<Word> = Vec::new();
        let mut merged = vec![false; words.len()];
        for (i, word) in words.iter().enumerate() {
            let compatible = i > 0
                && if boundary_only {
                    readings[i - 1].last().is_some_and(|p| p[1].ends_with('3'))
                        && readings[i].first().is_some_and(|p| p[1].ends_with('3'))
                } else {
                    i > 0 && third(&readings[i - 1]) && third(&readings[i])
                };
            if lexical(word)
                && compatible
                && !merged[i - 1]
                && !repeated(&words[i - 1].text)
                && words[i - 1].text.chars().count() + word.text.chars().count() <= 3
            {
                if let Some(previous) = output.last_mut() {
                    previous.text.push_str(&word.text);
                }
                merged[i] = true;
            } else {
                output.push(word.clone());
            }
        }
        Ok(output)
    }

    pub(super) fn modify_tones(&self, word: &Word, phones: &mut [[String; 2]]) -> Result<()> {
        let chars: Vec<_> = word.text.chars().collect();
        if chars.len() != phones.len() || phones.is_empty() {
            candle_core::bail!("misaligned Mandarin word pronunciation");
        }
        if chars.len() == 3 && chars[1] == '不' {
            tone(&mut phones[1], '5');
        } else {
            for i in 0..chars.len() - 1 {
                if chars[i] == '不' && phones[i + 1][1].ends_with('4') {
                    tone(&mut phones[i], '2');
                }
            }
        }
        let numeric_yi = chars.contains(&'一')
            && chars
                .iter()
                .filter(|&&c| c != '一')
                .all(|c| self.data.numeric_chars.contains(c));
        if !numeric_yi {
            if chars.len() == 3 && chars[1] == '一' && chars[0] == chars[2] {
                tone(&mut phones[1], '5');
            } else if word.text.starts_with("第一") {
                tone(&mut phones[1], '1');
            } else {
                for i in 0..chars.len() - 1 {
                    if chars[i] == '一' {
                        if phones[i + 1][1].ends_with(['4', '5']) {
                            tone(&mut phones[i], '2');
                        } else if !"、：，；。？！“”‘’':,;.?!".contains(chars[i + 1])
                        {
                            tone(&mut phones[i], '4');
                        }
                    }
                }
            }
        }
        self.neutralize(word, &chars, phones);
        match chars.len() {
            2 if third(phones) => tone(&mut phones[0], '2'),
            3 => {
                let (a, _) = self.split_word(&word.text);
                let split = a.chars().count();
                if third(phones) {
                    if split == 2 {
                        tone(&mut phones[0], '2');
                        tone(&mut phones[1], '2');
                    } else if split == 1 {
                        tone(&mut phones[1], '2');
                    }
                } else if split > 0 && split < phones.len() {
                    let (left, right) = phones.split_at_mut(split);
                    if left.len() == 2 && third(left) {
                        tone(&mut left[0], '2');
                    }
                    if right.len() == 2 && third(right) {
                        tone(&mut right[0], '2');
                    } else if !third(right)
                        && right[0][1].ends_with('3')
                        && left.last().is_some_and(|p| p[1].ends_with('3'))
                        && let Some(last) = left.last_mut()
                    {
                        tone(last, '2');
                    }
                }
            }
            4 => {
                for part in phones.chunks_mut(2) {
                    if third(part) {
                        tone(&mut part[0], '2');
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn neutralize(&self, word: &Word, chars: &[char], phones: &mut [[String; 2]]) {
        if self.data.not_neutral_words.contains(&word.text) {
            return;
        }
        for i in 1..chars.len() {
            if chars[i] == chars[i - 1] && word.tag.starts_with(['n', 'v', 'a']) {
                tone(&mut phones[i], '5');
            }
        }
        let last = chars.len() - 1;
        let neutral_last = "吧呢啊呐噻嘛吖嗨呐哦哒滴哩哟喽啰耶喔诶的地得".contains(chars[last])
            || (chars.len() == 1
                && "了着过".contains(chars[last])
                && ["ul", "uz", "ug"].contains(&word.tag.as_str()))
            || (chars.len() > 1
                && "们子".contains(chars[last])
                && ["r", "n"].contains(&word.tag.as_str()))
            || (chars.len() > 1
                && "上下".contains(chars[last])
                && ["s", "l", "f"].contains(&word.tag.as_str()))
            || (chars.len() > 1
                && "来去".contains(chars[last])
                && "上下进出回过起开".contains(chars[last - 1]));
        if neutral_last {
            tone(&mut phones[last], '5');
        } else if let Some(index) = chars.iter().position(|&c| c == '个').filter(|&i| {
            i > 0
                && (self.data.numeric_chars.contains(&chars[i - 1])
                    || "几有两半多各整每做是".contains(chars[i - 1]))
                || word.text == "个"
        }) {
            tone(&mut phones[index], '5');
        } else if self.data.neutral_words.contains(&word.text)
            || self.data.neutral_words.contains(&tail(&word.text, 2))
        {
            tone(&mut phones[last], '5');
        }
        let (a, b) = self.split_word(&word.text);
        let split = a.chars().count();
        for (text, start, end) in [(&a, 0, split), (&b, split, phones.len())] {
            if start < end
                && (self.data.neutral_words.contains(text)
                    || self.data.neutral_words.contains(&tail(text, 2)))
            {
                tone(&mut phones[end - 1], '5');
            }
        }
    }
}
