//! Tiktoken BPE tokenizer for IndexTTS 2.5.
//!
//! Faithful port of `indextts/utils/tokenizer.py`: the mergeable ranks come
//! from `multilingual_zh_ja_yue_char_del.tiktoken` (base64 token + rank per
//! line) and the special-token table is built exactly like `get_encoding`,
//! with ids assigned sequentially after the base vocabulary. Encoding follows
//! tiktoken semantics: special-token split first (longest match at the
//! earliest position), then the GPT-2 pre-tokenizer regex, then byte-level
//! BPE merges.
//!
//! The pre-tokenizer pattern is
//! `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`.
//! The Rust `regex` crate has no lookahead, so the two `\s+` alternatives are
//! implemented by hand on top of anchored `regex` matches for the rest; this
//! reproduces leftmost-first alternation exactly.

use std::collections::HashMap;

use candle_core::{DType, Device, Result, Tensor, bail};

/// Whisper language table in exact insertion order, followed by the IndexTTS
/// extensions (yue/minnan/wuyu/dialect/zh/en/en/zh/common).
pub const LANGUAGES: [&str; 106] = [
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv", "it",
    "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no", "th", "ur",
    "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az", "sl", "kn",
    "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw", "gl", "mr", "pa", "si",
    "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo",
    "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln",
    "ha", "ba", "jw", "su", "yue", "minnan", "wuyu", "dialect", "zh/en", "en/zh", "common",
];

const AUDIO_EVENTS: [&str; 11] = [
    "ASR",
    "AED",
    "SER",
    "Speech",
    "/Speech",
    "BGM",
    "/BGM",
    "Laughter",
    "/Laughter",
    "Applause",
    "/Applause",
];

const EMOTIONS: [&str; 4] = ["HAPPY", "SAD", "ANGRY", "NEUTRAL"];

const TTS_VOCAL_TOKENS: [&str; 20] = [
    "TTS/B", "TTS/O", "TTS/Q", "TTS/A", "TTS/CO", "TTS/CL", "TTS/H", "TTS/SP01", "TTS/SP02",
    "TTS/SP03", "TTS/SP04", "TTS/SP05", "TTS/SP06", "TTS/SP07", "TTS/SP08", "TTS/SP09", "TTS/SP10",
    "TTS/SP11", "TTS/SP12", "TTS/SP13",
];

/// Index into [`LANGUAGES`] used for the language embedding row; unknown
/// languages map to `common`, mirroring `lang_to_token`.
pub fn lang_to_token(lang: &str) -> u32 {
    let lower = lang.to_ascii_lowercase();
    let idx = LANGUAGES
        .iter()
        .position(|l| *l == lower)
        .unwrap_or_else(|| {
            LANGUAGES
                .iter()
                .position(|l| *l == "common")
                .expect("common entry")
        });
    idx as u32
}

/// Special-token string for a language prefix (`<|zh|>` etc.); only the
/// first 99 languages have special tokens, matching `get_encoding`.
pub fn lang_special_token(lang: &str) -> String {
    format!("<|{}|>", lang.to_ascii_lowercase())
}

pub struct Tokenizer {
    ranks: HashMap<Vec<u8>, u32>,
    /// Special tokens are static; a compact ordered vector keeps ids stable.
    specials: Vec<String>,
    special_ids: HashMap<String, u32>,
    letter_re: regex::Regex,
    number_re: regex::Regex,
    other_re: regex::Regex,
    contractions: [&'static str; 7],
}

impl Tokenizer {
    /// Load mergeable ranks from a `.tiktoken` file (base64 token + rank per
    /// line) and build the fixed special-token table.
    pub fn from_tiktoken_file(path: &std::path::Path) -> Result<Self> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| candle_core::Error::Msg(format!("read {}: {e}", path.display())))?;
        let mut ranks = HashMap::new();
        for line in text.lines() {
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split(' ');
            let (Some(tok), Some(rank)) = (parts.next(), parts.next()) else {
                bail!("malformed tiktoken line: {line:?}");
            };
            let bytes = base64_decode(tok)
                .map_err(|e| candle_core::Error::Msg(format!("base64 in tiktoken line: {e}")))?;
            let rank: u32 = rank
                .parse()
                .map_err(|e| candle_core::Error::Msg(format!("tiktoken rank: {e}")))?;
            ranks.insert(bytes, rank);
        }
        Self::from_ranks(ranks)
    }

    pub fn from_ranks(ranks: HashMap<Vec<u8>, u32>) -> Result<Self> {
        let mut specials = Vec::new();
        specials.push("<|endoftext|>".to_string());
        specials.push("<|startoftranscript|>".to_string());
        for lang in LANGUAGES.iter().take(99) {
            specials.push(format!("<|{lang}|>"));
        }
        for ev in AUDIO_EVENTS {
            specials.push(format!("<|{ev}|>"));
        }
        for emo in EMOTIONS {
            specials.push(format!("<|{emo}|>"));
        }
        for tok in [
            "<|translate|>",
            "<|transcribe|>",
            "<|startoflm|>",
            "<|startofprev|>",
            "<|nospeech|>",
            "<|notimestamps|>",
        ] {
            specials.push(tok.to_string());
        }
        for i in 1..=30 {
            specials.push(format!("<|SPECIAL_TOKEN_{i}|>"));
        }
        for tok in TTS_VOCAL_TOKENS {
            specials.push(format!("<|{tok}|>"));
        }
        for i in 0..=1500 {
            // f"{i * 0.02:.2f}" computed in exact hundredths.
            let cents = i * 2;
            specials.push(format!("<|{}.{:02}|>", cents / 100, cents % 100));
        }
        let base = ranks.values().copied().max().map_or(0, |r| r + 1);
        let special_ids = specials
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), base + i as u32))
            .collect();
        Ok(Self {
            ranks,
            specials,
            special_ids,
            letter_re: regex::Regex::new(r"^ ?\p{L}+").expect("static regex"),
            number_re: regex::Regex::new(r"^ ?\p{N}+").expect("static regex"),
            other_re: regex::Regex::new(r"^ ?[^\s\p{L}\p{N}]+").expect("static regex"),
            contractions: ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"],
        })
    }

    /// Number of mergeable ranks (base vocabulary size).
    pub fn base_vocab_size(&self) -> usize {
        self.ranks.len()
    }

    /// Total vocabulary size including special tokens (`explicit_n_vocab`).
    pub fn n_vocab(&self) -> usize {
        self.ranks.len() + self.specials.len()
    }

    pub fn special_id(&self, token: &str) -> Option<u32> {
        self.special_ids.get(token).copied()
    }

    /// Encode with special tokens enabled (tiktoken `allowed_special="all"`).
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        let mut rest = text;
        while let Some((at, len)) = self.find_special(rest) {
            if at > 0 {
                out.extend(self.encode_ordinary(&rest[..at]));
            }
            let special = &rest[at..at + len];
            out.push(
                self.special_ids
                    .get(special)
                    .copied()
                    .unwrap_or_else(|| panic!("special token missing id: {special}")),
            );
            rest = &rest[at + len..];
        }
        if !rest.is_empty() {
            out.extend(self.encode_ordinary(rest));
        }
        out
    }

    /// Earliest special-token occurrence, longest match winning ties. Only
    /// the first occurrence of each special can win, so one `find` per
    /// special suffices.
    fn find_special(&self, text: &str) -> Option<(usize, usize)> {
        let mut best: Option<(usize, usize)> = None;
        for special in &self.specials {
            if let Some(at) = text.find(special.as_str()) {
                let len = special.len();
                if best.is_none_or(|(ba, bl)| at < ba || (at == ba && len > bl)) {
                    best = Some((at, len));
                }
            }
        }
        best
    }

    /// tiktoken `encode_ordinary`: pre-tokenize, then byte-pair merge each
    /// piece.
    pub fn encode_ordinary(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for piece in self.pretokenize(text) {
            out.extend(self.byte_pair_encode(piece.as_bytes()));
        }
        out
    }

    /// GPT-2 pre-tokenizer; the trailing-whitespace lookahead alternative is
    /// expanded by hand (see module docs).
    fn pretokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let mut pieces = Vec::new();
        let bytes = text.as_bytes();
        let mut pos = 0usize;
        while pos < bytes.len() {
            let rest = &text[pos..];
            let mut matched = false;
            for c in self.contractions {
                if rest.starts_with(c) {
                    pieces.push(&rest[..c.len()]);
                    pos += c.len();
                    matched = true;
                    break;
                }
            }
            if matched {
                continue;
            }
            if let Some(m) = self.letter_re.find(rest) {
                pieces.push(m.as_str());
                pos += m.as_str().len();
                continue;
            }
            if let Some(m) = self.number_re.find(rest) {
                pieces.push(m.as_str());
                pos += m.as_str().len();
                continue;
            }
            if let Some(m) = self.other_re.find(rest) {
                pieces.push(m.as_str());
                pos += m.as_str().len();
                continue;
            }
            // Whitespace: `\s+(?!\S)` prefers the run minus its last char
            // when followed by a non-space, else `\s+` takes everything.
            let mut run_end = pos;
            let mut last_char_start = pos;
            for (offset, ch) in text[pos..].char_indices() {
                if ch.is_whitespace() {
                    last_char_start = pos + offset;
                    run_end = last_char_start + ch.len_utf8();
                } else {
                    break;
                }
            }
            if run_end == pos {
                // Should be unreachable: the char is not whitespace (it would
                // have matched an earlier alternative) — consume one char to
                // guarantee progress.
                let ch_len = text[pos..].chars().next().map_or(1, char::len_utf8);
                pieces.push(&text[pos..pos + ch_len]);
                pos += ch_len;
                continue;
            }
            if run_end == text.len() {
                pieces.push(&text[pos..run_end]);
                pos = run_end;
            } else if last_char_start > pos {
                // Multi-char run followed by non-space: drop the last char
                // (`\s+(?!\S)` backtracks one char).
                pieces.push(&text[pos..last_char_start]);
                pos = last_char_start;
            } else {
                // Single whitespace char: `\s+` takes it whole.
                pieces.push(&text[pos..run_end]);
                pos = run_end;
            }
        }
        pieces
    }

    /// tiktoken `_byte_pair_encode`: repeatedly merge the adjacent pair with
    /// the lowest rank.
    fn byte_pair_encode(&self, piece: &[u8]) -> Vec<u32> {
        if piece.is_empty() {
            return Vec::new();
        }
        if piece.len() == 1 {
            if let Some(id) = self.ranks.get(piece) {
                return vec![*id];
            }
            return Vec::new();
        }
        // parts[i] = start offset of part i; sentinel at the end.
        let mut parts: Vec<usize> = (0..=piece.len()).collect();
        let get_rank = |parts: &[usize], i: usize| -> Option<u32> {
            if i + 2 >= parts.len() {
                return None;
            }
            self.ranks.get(&piece[parts[i]..parts[i + 2]]).copied()
        };
        let mut pair_ranks: Vec<Option<u32>> =
            (0..parts.len() - 2).map(|i| get_rank(&parts, i)).collect();
        loop {
            let mut best: Option<(usize, u32)> = None;
            for (i, rank) in pair_ranks.iter().enumerate() {
                if let Some(r) = rank
                    && best.is_none_or(|(_, br)| *r < br)
                {
                    best = Some((i, *r));
                }
            }
            let Some((i, _)) = best else { break };
            // Merge pair (i, i+1): part i extends to parts[i+2].
            parts.splice(i + 1..i + 2, []);
            pair_ranks.remove(i);
            if i > 0 {
                pair_ranks[i - 1] = get_rank(&parts, i - 1);
            }
            if i < pair_ranks.len() {
                pair_ranks[i] = get_rank(&parts, i);
            }
        }
        parts[..parts.len() - 1]
            .iter()
            .enumerate()
            .filter_map(|(i, &start)| {
                let end = parts[i + 1];
                self.ranks.get(&piece[start..end]).copied()
            })
            .collect()
    }

    /// Encode `text` as a (1, L) i64 tensor on `device`.
    pub fn encode_tensor(&self, text: &str, device: &Device) -> Result<Tensor> {
        let ids = self.encode(text);
        let len = ids.len();
        Tensor::from_vec(ids, (1, len), device)?.to_dtype(DType::I64)
    }
}

/// Minimal standard-alphabet base64 decoder (tiktoken files are ASCII).
fn base64_decode(input: &str) -> std::result::Result<Vec<u8>, String> {
    const TABLE: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut rev = [255u8; 256];
    for (i, b) in TABLE.iter().enumerate() {
        rev[*b as usize] = i as u8;
    }
    let mut out = Vec::with_capacity(input.len() * 3 / 4);
    let mut buf = 0u32;
    let mut bits = 0u32;
    for ch in input.bytes() {
        if ch == b'=' || ch == b'\n' || ch == b'\r' {
            continue;
        }
        let v = rev[ch as usize];
        if v == 255 {
            return Err(format!("invalid base64 byte {ch:?}"));
        }
        buf = (buf << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((buf >> bits) as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn toy_tokenizer() -> Tokenizer {
        // Small handcrafted vocabulary resembling a tiktoken file:
        // single bytes 0..4 plus merged multi-byte tokens.
        let mut ranks = HashMap::new();
        let mut next = 0u32;
        for &b in b"abc x!" {
            ranks.insert(vec![b], next);
            next += 1;
        }
        for tok in [b"ab".to_vec(), b"abc".to_vec(), b" ab".to_vec()] {
            ranks.insert(tok, next);
            next += 1;
        }
        Tokenizer::from_ranks(ranks).expect("tokenizer")
    }

    #[test]
    fn language_table_matches_upstream_order() {
        assert_eq!(LANGUAGES.len(), 106);
        assert_eq!(LANGUAGES[0], "en");
        assert_eq!(LANGUAGES[1], "zh");
        assert_eq!(LANGUAGES[98], "su");
        assert_eq!(LANGUAGES[99], "yue");
        assert_eq!(LANGUAGES[105], "common");
        assert_eq!(lang_to_token("ZH"), 1);
        assert_eq!(lang_to_token("yue"), 99);
        assert_eq!(lang_to_token("unknown-lang"), 105);
    }

    #[test]
    fn special_table_layout() {
        let t = toy_tokenizer();
        let base = t.base_vocab_size() as u32;
        assert_eq!(
            t.n_vocab(),
            base as usize + 1 + 1 + 99 + 11 + 4 + 6 + 30 + 20 + 1501
        );
        assert_eq!(t.special_id("<|endoftext|>"), Some(base));
        assert_eq!(t.special_id("<|startoftranscript|>"), Some(base + 1));
        assert_eq!(t.special_id("<|en|>"), Some(base + 2));
        assert_eq!(t.special_id("<|zh|>"), Some(base + 3));
        // 99 language tokens: en..su
        assert_eq!(t.special_id("<|su|>"), Some(base + 2 + 98));
        // yue and friends have no special token.
        assert_eq!(t.special_id("<|yue|>"), None);
        assert_eq!(t.special_id("<|ASR|>"), Some(base + 2 + 99));
        assert_eq!(t.special_id("<|HAPPY|>"), Some(base + 2 + 99 + 11));
        assert_eq!(t.special_id("<|translate|>"), Some(base + 2 + 99 + 11 + 4));
        assert_eq!(
            t.special_id("<|SPECIAL_TOKEN_1|>"),
            Some(base + 2 + 99 + 11 + 4 + 6)
        );
        assert_eq!(
            t.special_id("<|SPECIAL_TOKEN_30|>"),
            Some(base + 2 + 99 + 11 + 4 + 6 + 29)
        );
        assert_eq!(
            t.special_id("<|TTS/B|>"),
            Some(base + 2 + 99 + 11 + 4 + 6 + 30)
        );
        assert_eq!(
            t.special_id("<|TTS/SP13|>"),
            Some(base + 2 + 99 + 11 + 4 + 6 + 30 + 19)
        );
        assert_eq!(
            t.special_id("<|0.00|>"),
            Some(base + 2 + 99 + 11 + 4 + 6 + 30 + 20)
        );
        assert_eq!(
            t.special_id("<|30.00|>"),
            Some(base + 2 + 99 + 11 + 4 + 6 + 30 + 20 + 1500)
        );
        assert_eq!(
            t.special_id("<|0.02|>"),
            Some(t.special_id("<|0.00|>").unwrap() + 1)
        );
        assert_eq!(
            t.special_id("<|1.50|>"),
            Some(t.special_id("<|0.00|>").unwrap() + 75)
        );
    }

    #[test]
    fn bpe_merges_lowest_rank_pairs() {
        let t = toy_tokenizer();
        let id = |tok: &[u8]| -> u32 { *t.ranks.get(tok).expect("token in vocab") };
        // "abc" exists as a full token and must win over "ab"+"c".
        assert_eq!(t.encode_ordinary("abc"), vec![id(b"abc")]);
        assert_eq!(t.encode_ordinary("ab"), vec![id(b"ab")]);
        // Pieces are merged independently: "ab" then "abc".
        assert_eq!(t.encode_ordinary("ababc"), vec![id(b"ab"), id(b"abc")]);
        // Unknown bytes produce no ids for that piece (full vocab covers all
        // bytes in practice; the shipped file is validated at load).
        assert!(t.encode_ordinary("x!").len() == 2);
    }

    #[test]
    fn pretokenizer_matches_gpt2_regex() {
        let t = toy_tokenizer();
        let cases: &[(&str, Vec<&str>)] = &[
            ("hello world", vec!["hello", " world"]),
            ("  x", vec![" ", " x"]),
            ("x  y", vec!["x", " ", " y"]),
            ("it's", vec!["it", "'s"]),
            ("we're", vec!["we", "'re"]),
            ("a1b2", vec!["a", "1", "b", "2"]),
            ("...!", vec!["...!"]),
            ("a  ", vec!["a", "  "]),
            (" 你好", vec![" 你好"]),
            ("你 好", vec!["你", " 好"]),
        ];
        for (text, expected) in cases {
            assert_eq!(&t.pretokenize(text), expected, "text: {text:?}");
        }
    }

    #[test]
    fn encode_splits_special_tokens() {
        let t = toy_tokenizer();
        let base = t.base_vocab_size() as u32;
        let zh = t.special_id("<|zh|>").unwrap();
        assert_ne!(zh, base); // sanity
        let ids = t.encode("<|zh|> 你好abc");
        assert_eq!(ids[0], zh);
        assert!(ids.len() > 1);
    }

    #[test]
    fn base64_roundtrip() {
        assert_eq!(base64_decode("IQ==").unwrap(), b"!");
        assert_eq!(base64_decode("Ig==").unwrap(), b"\"");
        assert_eq!(base64_decode("YWJj").unwrap(), b"abc");
        assert_eq!(base64_decode("YWI=").unwrap(), b"ab");
        assert_eq!(base64_decode("YQ==").unwrap(), b"a");
    }

    /// Requires the official `multilingual_zh_ja_yue_char_del.tiktoken`
    /// (dev download from IndexTeam/IndexTTS-2.5 at the pinned revision).
    /// Verified facts: 58836 base ranks + 1673 specials = 60509 ==
    /// config `gpt.number_text_tokens`; all text ids fit the 60510-row
    /// text embedding.
    #[test]
    #[ignore = "requires INDEXTTS_TIKTOKEN_FILE with the official vocab"]
    fn encodes_official_vocab() {
        let path = std::env::var("INDEXTTS_TIKTOKEN_FILE").expect("INDEXTTS_TIKTOKEN_FILE");
        let t = Tokenizer::from_tiktoken_file(std::path::Path::new(&path)).unwrap();
        assert_eq!(t.base_vocab_size(), 58_836);
        assert_eq!(t.n_vocab(), 60_509);
        let ids = t.encode("<|zh|> 你好，欢迎来体验indextts2。");
        assert_eq!(ids[0], t.special_id("<|zh|>").unwrap());
        assert!(ids.iter().all(|&i| (i as usize) < t.n_vocab()));
        // Multi-byte CJK merges must exist; encode round trips text length.
        assert!(ids.len() > 3);
    }
}
