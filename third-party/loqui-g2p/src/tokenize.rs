//! spaCy's English tokenizer, reimplemented.
//!
//! misaki's lexicon logic assumes spaCy's segmentation: "don't" arrives as
//! "do" + "n't", "$12.50" as "$" + "12.50", "Mr." stays whole. Matching it
//! exactly is what keeps loqui's phonemes identical to the ones Kokoro was
//! trained on, so this follows the control flow of spaCy's Cython
//! `Tokenizer` (`_tokenize_affixes`, `_split_affixes`, `_attach_tokens`)
//! step for step. The rules themselves (prefix, suffix, infix and URL
//! patterns, and 1,347 special cases) are data exported from spaCy 3.8 by
//! `tools/parity/spacy_export.py`. spaCy is MIT-licensed.

use std::collections::HashMap;

use fancy_regex::Regex;

use crate::Error;

static RULES: &str = include_str!("../data/spacy-en-tokenizer.json");

/// One token and whether a single space followed it in the source text.
/// Runs of extra whitespace become tokens of their own, as in spaCy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub text: String,
    pub space_after: bool,
}

pub struct Tokenizer {
    prefix: Regex,
    suffix: Regex,
    infix: Regex,
    url: Regex,
    specials: HashMap<String, Vec<String>>,
    /// Special cases that affix splitting can break apart ("):" becomes
    /// ")" + ":" when a word precedes it), keyed by the first piece of that
    /// broken form. See [`Tokenizer::merge_split_specials`].
    split_specials: HashMap<String, Vec<SplitSpecial>>,
}

/// A special case as it looks after affix splitting, and what it becomes.
struct SplitSpecial {
    pieces: Vec<String>,
    replacement: Vec<String>,
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer").field("specials", &self.specials.len()).finish_non_exhaustive()
    }
}

#[derive(serde::Deserialize)]
struct Rules {
    prefix: String,
    suffix: String,
    infix: String,
    url: String,
    special_cases: HashMap<String, Vec<String>>,
}

impl Tokenizer {
    /// The English rules of spaCy 3.8's `en_core_web_sm`.
    pub fn english() -> Result<Self, Error> {
        let rules: Rules = serde_json::from_str(RULES).map_err(|e| Error::Weights(format!("tokenizer rules: {e}")))?;
        let mut tokenizer = Self {
            prefix: compile(&rules.prefix)?,
            suffix: compile(&rules.suffix)?,
            infix: compile(&rules.infix)?,
            url: compile(&rules.url)?,
            specials: rules.special_cases,
            split_specials: HashMap::new(),
        };
        // spaCy registers a special case with its re-merging matcher when
        // the string contains an affix or a space, using the string's
        // tokenization with special cases switched off as the pattern.
        let mut split_specials: HashMap<String, Vec<SplitSpecial>> = HashMap::new();
        for (text, replacement) in &tokenizer.specials {
            let has_affix = text.contains(' ')
                || tokenizer.find_prefix(text) != 0
                || tokenizer.find_suffix(text) != 0
                || tokenizer.infix.find(text).ok().flatten().is_some();
            if !has_affix {
                continue;
            }
            let mut pieces = Vec::new();
            tokenizer.tokenize_span(text, &mut pieces, false);
            let pieces: Vec<String> = pieces.into_iter().map(|t| t.text).collect();
            if let Some(first) = pieces.first() {
                split_specials.entry(first.clone()).or_default().push(SplitSpecial { pieces, replacement: replacement.clone() });
            }
        }
        tokenizer.split_specials = split_specials;
        Ok(tokenizer)
    }

    pub fn tokenize(&self, text: &str) -> Vec<Token> {
        let mut tokens: Vec<Token> = Vec::new();
        let Some(first) = text.chars().next() else {
            return tokens;
        };
        // Like str.split, except that a span of whitespace is dropped only
        // when it is exactly one ' '. Longer runs keep all but that first
        // space as a token of their own.
        let mut in_ws = first.is_whitespace();
        let mut start = 0;
        for (i, c) in text.char_indices() {
            if c.is_whitespace() == in_ws {
                continue;
            }
            if start < i {
                self.tokenize_span(&text[start..i], &mut tokens, true);
            }
            if c == ' ' {
                if let Some(last) = tokens.last_mut() {
                    last.space_after = true;
                }
                start = i + 1;
            } else {
                start = i;
            }
            in_ws = !in_ws;
        }
        if start < text.len() {
            self.tokenize_span(&text[start..], &mut tokens, true);
            if let Some(last) = tokens.last_mut() {
                last.space_after = text.ends_with(' ') && !in_ws;
            }
        }
        self.merge_split_specials(tokens)
    }

    fn tokenize_span(&self, span: &str, out: &mut Vec<Token>, with_specials: bool) {
        if with_specials && self.push_special(span, out) {
            return;
        }
        let mut prefixes = Vec::new();
        let mut suffixes = Vec::new();
        let core = self.split_affixes(span, &mut prefixes, &mut suffixes, with_specials);
        out.extend(prefixes.into_iter().map(token));
        let special = !core.is_empty() && with_specials && self.push_special(core, out);
        if !core.is_empty() && !special {
            if self.url.is_match(core).unwrap_or(false) {
                out.push(token(core));
            } else {
                self.split_infixes(core, out);
            }
        }
        out.extend(suffixes.into_iter().rev().map(token));
    }

    /// Peels prefixes and suffixes off `span` until nothing more comes off
    /// or what remains is a special case. Mirrors `_split_affixes`.
    fn split_affixes<'a>(
        &self,
        mut span: &'a str,
        prefixes: &mut Vec<&'a str>,
        suffixes: &mut Vec<&'a str>,
        with_specials: bool,
    ) -> &'a str {
        let mut last_len = usize::MAX;
        while !span.is_empty() && span.len() != last_len {
            if with_specials && self.specials.contains_key(span) {
                break;
            }
            last_len = span.len();
            let pre_len = self.find_prefix(span);
            if pre_len != 0 {
                let minus_pre = &span[pre_len..];
                if with_specials && !minus_pre.is_empty() && self.specials.contains_key(minus_pre) {
                    prefixes.push(&span[..pre_len]);
                    span = minus_pre;
                    break;
                }
            }
            let suf_len = self.find_suffix(&span[pre_len..]);
            if suf_len != 0 {
                let minus_suf = &span[..span.len() - suf_len];
                if with_specials && !minus_suf.is_empty() && self.specials.contains_key(minus_suf) {
                    suffixes.push(&span[span.len() - suf_len..]);
                    span = minus_suf;
                    break;
                }
            }
            if pre_len != 0 && suf_len != 0 && pre_len + suf_len <= span.len() {
                prefixes.push(&span[..pre_len]);
                suffixes.push(&span[span.len() - suf_len..]);
                span = &span[pre_len..span.len() - suf_len];
            } else if pre_len != 0 {
                prefixes.push(&span[..pre_len]);
                span = &span[pre_len..];
            } else if suf_len != 0 {
                suffixes.push(&span[span.len() - suf_len..]);
                span = &span[..span.len() - suf_len];
            }
            // spaCy consults the special cases here even when they are
            // otherwise switched off.
            if !span.is_empty() && self.specials.contains_key(span) {
                break;
            }
        }
        span
    }

    /// Re-merges special cases that affix splitting broke apart: "bed):"
    /// peels ")" and ":" off separately, but "):" is a special case and
    /// ends up one token. Mirrors spaCy's `_apply_special_cases`, including
    /// its preference for the longest match and its refusal to match across
    /// whitespace.
    fn merge_split_specials(&self, tokens: Vec<Token>) -> Vec<Token> {
        // (start, end, replacement) for every place a pattern matches.
        let mut matches: Vec<(usize, usize, &[String])> = Vec::new();
        for (start, first) in tokens.iter().enumerate() {
            let Some(candidates) = self.split_specials.get(&first.text) else { continue };
            for special in candidates {
                let end = start + special.pieces.len();
                let Some(window) = tokens.get(start..end) else { continue };
                let joined = window[..window.len() - 1].iter().all(|t| !t.space_after);
                if joined && window.iter().zip(&special.pieces).all(|(t, p)| &t.text == p) {
                    matches.push((start, end, &special.replacement));
                }
            }
        }
        if matches.is_empty() {
            return tokens;
        }
        // Longest first; among equals, the later start first. A span is
        // kept only if neither of its end tokens is already claimed.
        matches.sort_by_key(|&(start, end, _)| (std::cmp::Reverse(end - start), std::cmp::Reverse(start)));
        let mut claimed = vec![false; tokens.len()];
        let mut chosen: Vec<(usize, usize, &[String])> = Vec::new();
        for (start, end, replacement) in matches {
            if !claimed[start] && !claimed[end - 1] {
                claimed[start..end].iter_mut().for_each(|c| *c = true);
                chosen.push((start, end, replacement));
            }
        }
        chosen.sort_by_key(|&(start, ..)| start);

        let mut out = Vec::with_capacity(tokens.len());
        let mut chosen = chosen.into_iter().peekable();
        let mut i = 0;
        while i < tokens.len() {
            match chosen.peek() {
                Some(&(start, end, replacement)) if start == i => {
                    out.extend(replacement.iter().map(|p| token(p)));
                    if let Some(last) = out.last_mut() {
                        last.space_after = tokens[end - 1].space_after;
                    }
                    chosen.next();
                    i = end;
                }
                _ => {
                    out.push(tokens[i].clone());
                    i += 1;
                }
            }
        }
        out
    }

    /// Splits on infixes (hyphens between letters, "..." and the like),
    /// keeping each infix as its own token. Mirrors `_attach_tokens`.
    fn split_infixes(&self, core: &str, out: &mut Vec<Token>) {
        let mut start = 0;
        for m in self.infix.find_iter(core).filter_map(Result::ok) {
            if m.start() == 0 {
                continue;
            }
            if m.start() != start {
                out.push(token(&core[start..m.start()]));
            }
            if m.start() != m.end() {
                out.push(token(m.as_str()));
            }
            start = m.end();
        }
        if start < core.len() {
            out.push(token(&core[start..]));
        }
    }

    /// Whether `text` is a URL or e-mail address by spaCy's own pattern:
    /// "example.com", "docs.rs/loqui", "ana@example.com".
    pub fn is_url(&self, text: &str) -> bool {
        text.contains('.') && self.url.is_match(text).unwrap_or(false)
    }

    fn push_special(&self, span: &str, out: &mut Vec<Token>) -> bool {
        match self.specials.get(span) {
            Some(pieces) => {
                out.extend(pieces.iter().map(|p| token(p)));
                true
            }
            None => false,
        }
    }

    fn find_prefix(&self, s: &str) -> usize {
        self.prefix.find(s).ok().flatten().map_or(0, |m| m.end() - m.start())
    }

    fn find_suffix(&self, s: &str) -> usize {
        self.suffix.find(s).ok().flatten().map_or(0, |m| m.end() - m.start())
    }
}

fn token(text: &str) -> Token {
    Token { text: text.to_owned(), space_after: false }
}

/// Compiles a pattern written for Python's `re`. Two dialect differences
/// matter: Python tolerates escaping quote characters, and spaCy's URL
/// pattern opens with a `(?u)` flag that is already the default here.
fn compile(pattern: &str) -> Result<Regex, Error> {
    let pattern = pattern.replace("(?u)", "").replace("\\'", "'").replace("\\\"", "\"");
    Regex::new(&pattern).map_err(|e| Error::Weights(format!("tokenizer pattern: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texts(tokens: &[Token]) -> Vec<&str> {
        tokens.iter().map(|t| t.text.as_str()).collect()
    }

    /// Each expectation is spaCy 3.8's own output for the same string.
    #[test]
    fn matches_spacy_on_the_cases_misaki_depends_on() {
        let t = Tokenizer::english().unwrap();
        let cases: &[(&str, &[&str])] = &[
            ("$12.50 a month", &["$", "12.50", "a", "month"]),
            ("I've read it, don't you?", &["I", "'ve", "read", "it", ",", "do", "n't", "you", "?"]),
            ("March 3rd, 2027.", &["March", "3rd", ",", "2027", "."]),
            ("9:30 AM", &["9:30", "AM"]),
            ("73%", &["73", "%"]),
            ("example.com/docs", &["example.com/docs"]),
            ("Mr. Smith", &["Mr.", "Smith"]),
            ("Wi-Fi café", &["Wi", "-", "Fi", "café"]),
            ("-5 degrees", &["-5", "degrees"]),
            ("$1,234,567.89", &["$", "1,234,567.89"]),
            ("one-on-one", &["one", "-", "on", "-", "one"]),
            ("#115", &["#", "115"]),
            ("10/14", &["10/14"]),
            ("555-0123", &["555", "-", "0123"]),
            ("bed):", &["bed", "):"]),
            ("bed) :", &["bed", ")", ":"]),
            ("—Mr. Rochester", &["—", "Mr.", "Rochester"]),
            ("Mr . Smith", &["Mr", ".", "Smith"]),
            ("find ’em.", &["find", "’em", "."]),
            ("a-doin’ well", &["a", "-", "doin’", "well"]),
        ];
        for (text, expected) in cases {
            assert_eq!(texts(&t.tokenize(text)), *expected, "tokenizing {text:?}");
        }
    }

    #[test]
    fn records_single_spaces_and_keeps_longer_runs() {
        let t = Tokenizer::english().unwrap();
        let tokens = t.tokenize("a  b ");
        assert_eq!(texts(&tokens), ["a", " ", "b"]);
        assert_eq!(tokens.iter().map(|t| t.space_after).collect::<Vec<_>>(), [true, false, true]);
    }
}
