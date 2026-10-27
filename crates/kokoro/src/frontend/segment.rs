//! jieba 0.42.1 POS-HMM segmentation, preserving the official frontend's word boundaries.
use super::{Frontend, Word};
use candle_core::Result;
use serde::Deserialize;
use std::collections::{HashMap, HashSet};

#[derive(Deserialize)]
pub(super) struct PosHmm {
    states: Vec<[String; 2]>,
    start: Vec<f64>,
    transitions: Vec<HashMap<usize, f64>>,
    emissions: Vec<HashMap<char, f64>>,
    char_states: HashMap<char, Vec<usize>>,
}

impl PosHmm {
    fn cut(&self, text: &str) -> Result<Vec<Word>> {
        let chars: Vec<_> = text.chars().collect();
        if chars.is_empty() {
            return Ok(Vec::new());
        }
        let count = self.states.len();
        if count == 0
            || self.start.len() != count
            || self.transitions.len() != count
            || self.emissions.len() != count
        {
            candle_core::bail!("invalid official Mandarin POS-HMM data");
        }
        let all: Vec<_> = (0..count).collect();
        let mut active = self.char_states.get(&chars[0]).unwrap_or(&all).clone();
        let mut previous = vec![f64::NEG_INFINITY; count];
        let mut paths = vec![vec![usize::MAX; count]; chars.len()];
        for &state in &active {
            if state >= count {
                candle_core::bail!("invalid Mandarin POS state");
            }
            previous[state] = self.start[state]
                + self.emissions[state]
                    .get(&chars[0])
                    .copied()
                    .unwrap_or(-3.14e100);
        }
        for (position, &character) in chars.iter().enumerate().skip(1) {
            active.retain(|&state| !self.transitions[state].is_empty());
            let expected: HashSet<_> = active
                .iter()
                .flat_map(|&state| self.transitions[state].keys().copied())
                .collect();
            let mut observed: Vec<_> = self
                .char_states
                .get(&character)
                .unwrap_or(&all)
                .iter()
                .copied()
                .filter(|state| expected.contains(state))
                .collect();
            if observed.is_empty() {
                observed = if expected.is_empty() {
                    all.clone()
                } else {
                    expected.into_iter().collect()
                };
            }
            let mut current = vec![f64::NEG_INFINITY; count];
            for &state in &observed {
                if state >= count {
                    candle_core::bail!("invalid Mandarin POS transition");
                }
                let emission = self.emissions[state]
                    .get(&character)
                    .copied()
                    .unwrap_or(-3.14e100);
                let best = active
                    .iter()
                    .map(|&from| {
                        (
                            previous[from]
                                + self.transitions[from]
                                    .get(&state)
                                    .copied()
                                    .unwrap_or(f64::NEG_INFINITY)
                                + emission,
                            from,
                        )
                    })
                    .max_by(|(a, i), (b, j)| a.total_cmp(b).then(i.cmp(j)));
                let Some((score, from)) = best else {
                    candle_core::bail!("Mandarin POS-HMM has no valid predecessor");
                };
                current[state] = score;
                paths[position][state] = from;
            }
            previous = current;
            active = observed;
        }
        let mut state = active
            .into_iter()
            .max_by(|&a, &b| previous[a].total_cmp(&previous[b]).then(a.cmp(&b)))
            .ok_or_else(|| {
                candle_core::Error::Msg("Mandarin POS-HMM has no terminal state".into())
            })?;
        let mut route = vec![0; chars.len()];
        for position in (0..chars.len()).rev() {
            route[position] = state;
            if position > 0 {
                state = paths[position][state];
                if state == usize::MAX {
                    candle_core::bail!("broken Mandarin POS backtrace");
                }
            }
        }
        let mut words = Vec::new();
        let mut begin = 0;
        let mut next = 0;
        for (i, &state) in route.iter().enumerate() {
            match self.states[state][0].as_str() {
                "B" => begin = i,
                "E" => {
                    words.push(Word {
                        text: chars[begin..=i].iter().collect(),
                        tag: self.states[state][1].clone(),
                    });
                    next = i + 1;
                }
                "S" => {
                    words.push(Word {
                        text: chars[i].to_string(),
                        tag: self.states[state][1].clone(),
                    });
                    next = i + 1;
                }
                _ => {}
            }
        }
        if next < chars.len() {
            words.push(Word {
                text: chars[next..].iter().collect(),
                tag: self.states[route[next]][1].clone(),
            });
        }
        Ok(words)
    }
}

impl Frontend {
    fn dictionary_word(&self, text: &str) -> Word {
        Word {
            text: text.into(),
            tag: self
                .tags
                .get(text)
                .map_or("x", |(_, tag)| tag.as_str())
                .into(),
        }
    }

    fn flush_single_buffer(&self, buffer: &str, words: &mut Vec<Word>) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }
        if buffer.chars().count() == 1 {
            words.push(self.dictionary_word(buffer));
        } else if self.tags.get(buffer).is_some_and(|(freq, _)| *freq > 0) {
            words.extend(buffer.chars().map(|c| self.dictionary_word(&c.to_string())));
        } else {
            words.extend(self.data.pos_hmm.cut(buffer)?);
        }
        Ok(())
    }

    pub(super) fn pos_segment(&self, text: &str) -> Result<Vec<Word>> {
        let mut words = Vec::new();
        let mut buffer = String::new();
        // The DAG is supplied by Jieba, but OOV single-character runs use the
        // official POS-HMM, not the distinct four-state word-only HMM.
        for token in self.jieba.cut(text, false) {
            let han = token
                .word
                .chars()
                .all(|c| ('\u{4e00}'..='\u{9fd5}').contains(&c));
            if han && token.word.chars().count() == 1 {
                buffer.push_str(token.word);
                continue;
            }
            self.flush_single_buffer(&buffer, &mut words)?;
            buffer.clear();
            if han {
                words.push(self.dictionary_word(token.word));
            } else {
                for c in token.word.chars() {
                    words.push(Word {
                        text: c.to_string(),
                        tag: "x".into(),
                    });
                }
            }
        }
        self.flush_single_buffer(&buffer, &mut words)?;
        Ok(words)
    }
}
