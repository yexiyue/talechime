//! Chinese/English text frontend (`indextts/utils/front.py` subset).
//!
//! Ported exactly: `char_rep_map`/`zh_char_rep_map` first-match folding,
//! `<字|读音>` pronunciation annotations, pinyin-tone save/restore with
//! `correct_pinyin` (jqx + u/ü → v), tech-term hyphen protection, name
//! protection, the English-contraction rewrite and `use_chinese`
//! language routing.
//!
//! Documented approximation: upstream `normalize()` delegates number/date
//! normalization to the wetext / TensorFlow-TN stack, which is not ported.
//! This module implements a pragmatic subset (Chinese and English
//! cardinal numbers, decimals, percentages, dates, clock times). Model
//! modules downstream of `normalize` are exact; this stage is not, and
//! Python-oracle comparisons must exclude it.

use std::collections::HashMap;

use regex::Regex;

/// (pattern, replacement) pairs in upstream insertion order. Replacement
/// scans position by position and takes the first matching key, exactly
/// like Python's `re.compile("|".join(keys)).sub(...)` — note this means
/// the later multi-char keys `...`/`,,,`/`，，，` can be shadowed by
/// earlier single-char keys, faithfully to upstream.
const CHAR_REP_MAP: [(&str, &str); 31] = [
    ("：", ","),
    ("；", ","),
    (";", ","),
    ("，", ","),
    ("。", "."),
    ("！", "!"),
    ("？", "?"),
    ("\n", " "),
    ("·", "-"),
    ("、", ","),
    ("...", "…"),
    (",,,", "…"),
    ("，，，", "…"),
    ("……", "…"),
    ("“", "'"),
    ("”", "'"),
    ("\"", "'"),
    ("‘", "'"),
    ("’", "'"),
    ("（", "'"),
    ("）", "'"),
    ("(", "'"),
    (")", "'"),
    ("《", "'"),
    ("》", "'"),
    ("【", "'"),
    ("】", "'"),
    ("[", "'"),
    ("]", "'"),
    ("—", "-"),
    ("～", "-"),
];

/// zh branch adds these before the shared map (`zh_char_rep_map`).
const ZH_EXTRA_MAP: [(&str, &str); 1] = [("$", ".")];

/// The remaining entries of the full maps (`~`, `「`, `」`, `:`).
const CHAR_REP_MAP_TAIL: [(&str, &str); 4] = [("~", "-"), ("「", "'"), ("」", "'"), (":", ",")];

pub struct TextFrontend {
    glossary: Vec<(String, String)>,
    zh_map: Vec<(&'static str, &'static str)>,
    base_map: Vec<(&'static str, &'static str)>,
    contraction_re: Regex,
    tech_re: Regex,
    name_re: Regex,
    email_re: Regex,
    chinese_re: Regex,
    alpha_re: Regex,
    pinyin_re: Regex,
    pron_re: Regex,
}

impl TextFrontend {
    pub fn new() -> Self {
        let mut zh_map =
            Vec::with_capacity(ZH_EXTRA_MAP.len() + CHAR_REP_MAP.len() + CHAR_REP_MAP_TAIL.len());
        zh_map.extend_from_slice(&ZH_EXTRA_MAP);
        zh_map.extend_from_slice(&CHAR_REP_MAP);
        zh_map.extend_from_slice(&CHAR_REP_MAP_TAIL);
        let mut base_map = Vec::with_capacity(CHAR_REP_MAP.len() + CHAR_REP_MAP_TAIL.len());
        base_map.extend_from_slice(&CHAR_REP_MAP);
        base_map.extend_from_slice(&CHAR_REP_MAP_TAIL);
        Self {
            glossary: Vec::new(),
            zh_map,
            base_map,
            // (?i)(what|where|...)'s -> \1 is
            contraction_re: Regex::new(
                r"(?i)(what|where|who|which|how|t?here|it|s?he|that|this)'s",
            )
            .expect("static regex"),
            tech_re: Regex::new(r"[A-Za-z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)+").expect("static regex"),
            name_re: Regex::new(r"[\x{4e00}-\x{9fff}]+(?:[-·—][\x{4e00}-\x{9fff}]+){1,2}")
                .expect("static regex"),
            email_re: Regex::new(r"^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-zA-Z]+$").expect("static regex"),
            chinese_re: Regex::new(r"[\x{4e00}-\x{9fff}]").expect("static regex"),
            alpha_re: Regex::new(r"[a-zA-Z]").expect("static regex"),
            // PINYIN_TONE_PATTERN without its leading lookbehind; the check
            // of the preceding character is done by the caller.
            pinyin_re: Regex::new(
                r"(?i)((?:[bpmfdtnlgkhjqxzcsryw]|[zcs]h)?(?:[aeiouüv]|[ae]i|u[aio]|ao|ou|i[aue]|[uüv]e|[uvü]ang?|uai|[aeiuv]n|[aeio]ng|ia[no]|i[ao]ng)|ng|er)([1-5])",
            )
            .expect("static regex"),
            pron_re: Regex::new(r"<([^|>\n]+)\|([^>\n]+)>").expect("static regex"),
        }
    }

    /// Load a glossary (term → reading) applied before all protections,
    /// longest term first, case-insensitive (`apply_glossary_terms`).
    pub fn set_glossary(&mut self, glossary: HashMap<String, String>) {
        let mut entries: Vec<(String, String)> = glossary.into_iter().collect();
        entries.sort_by_key(|(term, _)| std::cmp::Reverse(term.len()));
        self.glossary = entries;
    }

    /// The `clean_pattern` substitution applied by `infer_generator`
    /// (`char_rep_map` folding) before normalization.
    pub fn clean_pattern(&self, text: &str) -> String {
        self.fold(text, &self.base_map)
    }

    fn fold(&self, text: &str, map: &[(&str, &str)]) -> String {
        let mut out = String::with_capacity(text.len());
        let mut rest = text;
        'outer: while !rest.is_empty() {
            for (pat, rep) in map {
                if rest.starts_with(pat) {
                    out.push_str(rep);
                    rest = &rest[pat.len()..];
                    continue 'outer;
                }
            }
            let ch = rest.chars().next().expect("non-empty");
            out.push(ch);
            rest = &rest[ch.len_utf8()..];
        }
        out
    }

    fn use_chinese(&self, s: &str) -> bool {
        let has_chinese = self.chinese_re.is_match(s);
        let has_alpha = self.alpha_re.is_match(s);
        let is_email = self.email_re.is_match(s);
        if has_chinese || !has_alpha || is_email {
            return true;
        }
        self.has_pinyin_tone(s)
    }

    fn has_pinyin_tone(&self, s: &str) -> bool {
        for m in self.pinyin_re.find_iter(s) {
            let byte_start = m.start();
            if !self.prev_is_ascii_letter(s, byte_start) {
                return true;
            }
        }
        false
    }

    fn prev_is_ascii_letter(&self, s: &str, byte_start: usize) -> bool {
        s[..byte_start]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_alphabetic())
    }

    /// `TextNormalizer.normalize`. The zh/en branch is chosen with
    /// `use_chinese` unless `lang_zh` forces it.
    pub fn normalize(&self, text: &str, lang_zh: bool) -> String {
        let zh = lang_zh || self.use_chinese(text);
        // Protect <字|读音> annotations from all rewriting.
        let mut placeholders: Vec<(String, String)> = Vec::new();
        let mut protected = self
            .pron_re
            .replace_all(text, |caps: &regex::Captures| {
                let tag = alpha_index(placeholders.len());
                let key = format!("PRONPLACEHOLDER{tag}PRONPLACEHOLDER");
                placeholders.push((key.clone(), caps[0].to_string()));
                key
            })
            .to_string();

        let contraction_ok = self
            .contraction_re
            .replace_all(&protected, "${1} is")
            .to_string();
        protected = contraction_ok;

        if !self.glossary.is_empty() {
            let mut t = protected;
            for (term, reading) in &self.glossary {
                let re = Regex::new(&regex::escape(term)).expect("escaped glossary term");
                t = re.replace_all(&t, reading.as_str()).to_string();
            }
            protected = t;
        }

        let result = if zh {
            let (step1, tech) = self.save_tech_terms(&protected);
            let (step2, pinyin) = self.save_pinyin_tones(&step1);
            let (step3, names) = self.save_names(&step2);
            let mut normalized = zh_number_normalize(&step3);
            if let Some(names) = names {
                normalized = restore_placeholders_generic(&normalized, &names, "<n_");
            }
            if let Some(pinyin) = pinyin {
                normalized = self.restore_pinyin_tones(&normalized, &pinyin);
            }
            let normalized = self.restore_tech_terms(normalized, &tech);
            self.fold(&normalized, &self.zh_map)
        } else {
            let (step1, tech) = self.save_tech_terms(&protected);
            let normalized = en_number_normalize(&step1);
            let normalized = self.restore_tech_terms(normalized, &tech);
            self.fold(&normalized, &self.base_map)
        };

        let mut out = result;
        for (key, original) in placeholders {
            out = out.replace(&key, &original);
        }
        out
    }

    fn save_tech_terms(&self, text: &str) -> (String, Vec<String>) {
        let mut terms: Vec<String> = self
            .tech_re
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect();
        terms.sort();
        terms.dedup();
        terms.sort_by_key(|t| std::cmp::Reverse(t.len()));
        let mut out = text.to_string();
        for term in &terms {
            let protected = term.replace('-', "<H>");
            out = out.replace(term, &protected);
        }
        (out, terms)
    }

    fn restore_tech_terms(&self, text: String, _terms: &[String]) -> String {
        // `\s*<H>\s*` -> '-'
        let mut out = String::with_capacity(text.len());
        let bytes = text.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            if text[i..].starts_with("<H>") {
                // consume surrounding whitespace already skipped below
                out.push('-');
                i += 3;
                while i < bytes.len() && (bytes[i] as char).is_ascii_whitespace() {
                    i += 1;
                }
            } else {
                let ch_len = text[i..].chars().next().map_or(1, char::len_utf8);
                if (bytes[i] as char).is_ascii_whitespace() {
                    let j = i + ch_len;
                    if text[j..].starts_with("<H>") {
                        i = j; // drop the whitespace, loop handles <H>
                        continue;
                    }
                }
                out.push_str(&text[i..i + ch_len]);
                i += ch_len;
            }
        }
        out
    }

    fn save_pinyin_tones(&self, text: &str) -> (String, Option<Vec<String>>) {
        let mut found: Vec<String> = Vec::new();
        for m in self.pinyin_re.find_iter(text) {
            if self.prev_is_ascii_letter(text, m.start()) {
                continue;
            }
            found.push(m.as_str().to_string());
        }
        if found.is_empty() {
            return (text.to_string(), None);
        }
        found.sort();
        found.dedup();
        let mut out = text.to_string();
        for (i, pinyin) in found.iter().enumerate() {
            let letter = alpha_index(i);
            out = out.replace(pinyin, &format!("<pinyin_{letter}>"));
        }
        (out, Some(found))
    }

    fn restore_pinyin_tones(&self, text: &str, pinyins: &[String]) -> String {
        let mut out = text.to_string();
        for (i, pinyin) in pinyins.iter().enumerate() {
            let letter = alpha_index(i);
            out = out.replace(&format!("<pinyin_{letter}>"), &correct_pinyin(pinyin));
        }
        out
    }

    fn save_names(&self, text: &str) -> (String, Option<Vec<String>>) {
        let mut names: Vec<String> = self
            .name_re
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect();
        if names.is_empty() {
            return (text.to_string(), None);
        }
        names.sort();
        names.dedup();
        let mut out = text.to_string();
        for (i, name) in names.iter().enumerate() {
            let letter = alpha_index(i);
            out = out.replace(name, &format!("<n_{letter}>"));
        }
        (out, Some(names))
    }
}

impl Default for TextFrontend {
    fn default() -> Self {
        Self::new()
    }
}

/// `_idx_to_alpha`: 0→a, 25→z, 26→aa, ... (upstream only ever needs a few).
fn alpha_index(i: usize) -> String {
    let mut n = i;
    let mut s = String::new();
    loop {
        s.insert(0, (b'a' + (n % 26) as u8) as char);
        if n < 26 {
            return s;
        }
        n = n / 26 - 1;
    }
}

fn restore_placeholders_generic(text: &str, items: &[String], prefix: &str) -> String {
    let mut out = text.to_string();
    for (i, item) in items.iter().enumerate() {
        let letter = alpha_index(i);
        out = out.replace(&format!("{prefix}{letter}>"), item);
    }
    out
}

/// `correct_pinyin`: jqx + u/ü rhyme → v, then uppercase.
fn correct_pinyin(pinyin: &str) -> String {
    let first = pinyin.chars().next().unwrap_or(' ');
    if !"jqxJQX".contains(first) {
        return pinyin.to_string();
    }
    let re = Regex::new(r"(?i)([jqx])[uü](n|e|an)*(\d)").expect("static regex");
    re.replace(pinyin, "${1}v${2}${3}").to_uppercase()
}

/// `<文字|发音>` → `<|SPECIAL_TOKEN_1|>/<|SPECIAL_TOKEN_2|>` wrapping with
/// kana passthrough (`apply_pronunciation_annotations` in `infer_v2_5.py`).
pub fn apply_pronunciation_annotations(text: &str) -> String {
    let pron_re = Regex::new(r"<([^|>\n]+)\|([^>\n]+)>").expect("static regex");
    let hira = Regex::new(r"^[\x{3040}-\x{309F}]+$").expect("static regex");
    let kata = Regex::new(r"^[\x{30A0}-\x{30FF}]+$").expect("static regex");
    let chinese = Regex::new(r"[\x{4e00}-\x{9fff}]").expect("static regex");
    pron_re
        .replace_all(text, |caps: &regex::Captures| {
            let word = caps.get(1).expect("group 1").as_str();
            let pronunciation = caps.get(2).expect("group 2").as_str().to_uppercase();
            if hira.is_match(&pronunciation) || kata.is_match(&pronunciation) {
                format!(" {pronunciation} ")
            } else {
                let token = if chinese.is_match(word) {
                    "SPECIAL_TOKEN_2"
                } else {
                    "SPECIAL_TOKEN_1"
                };
                format!("<|{token}|>{pronunciation}<|{token}|>")
            }
        })
        .to_string()
}

// ---------------------------------------------------------------------------
// Number normalization subset (the documented approximation).
// ---------------------------------------------------------------------------

const CN_DIGITS: [&str; 10] = ["零", "一", "二", "三", "四", "五", "六", "七", "八", "九"];
const CN_UNITS: [(&str, u32); 4] = [("十", 10), ("百", 100), ("千", 1_000), ("万", 10_000)];

/// Integer ≤ 99_999_999 to Chinese cardinal reading.
fn chinese_cardinal(mut n: u64) -> String {
    if n == 0 {
        return "零".to_string();
    }
    if n >= 100_000_000 {
        // 亿-scale; recurse once.
        let yi = n / 100_000_000;
        n %= 100_000_000;
        let rest = if n > 0 {
            chinese_cardinal(n)
        } else {
            String::new()
        };
        let head = if rest.is_empty() {
            format!("{}亿", chinese_cardinal(yi))
        } else if n < 10_000_000 {
            format!("{}亿零{}", chinese_cardinal(yi), rest)
        } else {
            format!("{}亿{}", chinese_cardinal(yi), rest)
        };
        return head;
    }
    let mut parts: Vec<String> = Vec::new();
    let wan = n / 10_000;
    let low = n % 10_000;
    if wan > 0 {
        parts.push(format!("{}万", four_digits(wan as u32)));
    }
    if low > 0 {
        let s = if wan > 0 && low < 1000 {
            format!("零{}", four_digits(low as u32))
        } else {
            four_digits(low as u32)
        };
        parts.push(s);
    }
    parts.concat()
}

/// 1..=9999 with 十/百/千 units and interior 零.
fn four_digits(mut n: u32) -> String {
    let mut out = String::new();
    let mut zero_pending = false;
    for &(unit, scale) in CN_UNITS.iter().rev() {
        let d = n / scale;
        n %= scale;
        if d == 0 {
            if !out.is_empty() && n > 0 {
                zero_pending = true;
            }
            continue;
        }
        if zero_pending {
            out.push('零');
            zero_pending = false;
        }
        if !(d == 1 && unit == "十" && out.is_empty()) {
            out.push_str(CN_DIGITS[d as usize]);
        }
        out.push_str(unit);
    }
    if n > 0 {
        if zero_pending {
            out.push('零');
        }
        out.push_str(CN_DIGITS[n as usize]);
    }
    out
}

fn zh_number_normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    let digit = |c: char| c.is_ascii_digit();
    while i < n {
        let c = chars[i];
        if !digit(c) {
            out.push(c);
            i += 1;
            continue;
        }
        // Consume [0-9]+(\.[0-9]+)?
        let start = i;
        let mut j = i;
        while j < n && digit(chars[j]) {
            j += 1;
        }
        let mut decimal_digits: Vec<u32> = Vec::new();
        if j + 1 < n && chars[j] == '.' && digit(chars[j + 1]) {
            let mut k = j + 1;
            while k < n && digit(chars[k]) {
                decimal_digits.push(chars[k].to_digit(10).expect("ascii digit"));
                k += 1;
            }
            j = k;
        }
        let int_digits: Vec<u32> = {
            let mut v = Vec::new();
            let mut k = start;
            while k < n && digit(chars[k]) {
                v.push(chars[k].to_digit(10).expect("ascii digit"));
                k += 1;
            }
            v
        };
        let int_value: u64 = int_digits.iter().fold(0u64, |acc, d| acc * 10 + *d as u64);

        // Year pattern: 4 digits followed by 年 → digit-wise reading.
        if int_digits.len() == 4 && j < n && chars[j] == '年' {
            for d in &int_digits {
                out.push_str(CN_DIGITS[*d as usize]);
            }
            out.push('年');
            i = j + 1;
            continue;
        }
        // Date units: read as cardinal.
        let unit = if j < n { Some(chars[j]) } else { None };
        let reading = if unit == Some('月') || unit == Some('日') || unit == Some('号') {
            // Month/day: cardinals, but 10..19 read 十X (four_digits does).
            four_digits(int_value.min(99) as u32)
        } else {
            let mut s = chinese_cardinal(int_value);
            if !decimal_digits.is_empty() {
                s.push('点');
                for d in &decimal_digits {
                    s.push_str(CN_DIGITS[*d as usize]);
                }
            }
            s
        };
        if unit == Some('%') {
            out.push_str("百分之");
            out.push_str(&reading);
            i = j + 1;
            continue;
        }
        out.push_str(&reading);
        i = j;
    }
    out
}

const EN_ONES: [&str; 20] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
];
const EN_TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];

fn en_below_1000(n: u64) -> String {
    let mut out = String::new();
    let hundreds = n / 100;
    let mut rest = n % 100;
    if hundreds > 0 {
        out.push_str(EN_ONES[hundreds as usize]);
        out.push_str(" hundred");
    }
    if rest > 0 {
        if !out.is_empty() {
            out.push(' ');
        }
        if rest < 20 {
            out.push_str(EN_ONES[rest as usize]);
        } else {
            out.push_str(EN_TENS[(rest / 10) as usize]);
            rest %= 10;
            if rest > 0 {
                out.push('-');
                out.push_str(EN_ONES[rest as usize]);
            }
        }
    }
    out
}

fn english_cardinal(n: u64) -> String {
    if n < 1000 {
        return en_below_1000(n);
    }
    let mut parts: Vec<String> = Vec::new();
    let scales: [(u64, &str); 3] = [
        (1_000_000_000, "billion"),
        (1_000_000, "million"),
        (1_000, "thousand"),
    ];
    let mut rest = n;
    for (scale, name) in scales {
        let v = rest / scale;
        rest %= scale;
        if v > 0 {
            parts.push(format!("{} {}", en_below_1000(v), name));
        }
    }
    if rest > 0 {
        parts.push(en_below_1000(rest));
    }
    parts.join(" ")
}

fn en_number_normalize(text: &str) -> String {
    let re = Regex::new(r"[0-9]+(\.[0-9]+)?").expect("static regex");
    re.replace_all(text, |caps: &regex::Captures| {
        let whole = caps.get(0).expect("whole match").as_str();
        let (int_part, frac_part) = match whole.split_once('.') {
            Some((i, f)) => (i, Some(f)),
            None => (whole, None),
        };
        let int_value: u64 = int_part.parse().unwrap_or(0);
        let mut s = english_cardinal(int_value);
        if let Some(frac) = frac_part {
            s.push_str(" point");
            for c in frac.chars() {
                let d = c.to_digit(10).expect("ascii digit") as usize;
                s.push(' ');
                s.push_str(EN_ONES[d]);
            }
        }
        s
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_pattern_folds_punctuation() {
        let f = TextFrontend::new();
        assert_eq!(f.clean_pattern("你好，世界。"), "你好,世界.");
        assert_eq!(f.clean_pattern("“引号”"), "'引号'");
        assert_eq!(f.clean_pattern("第一\n行"), "第一 行");
    }

    #[test]
    fn pinyin_tone_detection() {
        let f = TextFrontend::new();
        assert!(f.has_pinyin_tone("晕XUAN4是一种GAN3觉"));
        assert!(f.has_pinyin_tone("zhong4yao4"));
        assert!(!f.has_pinyin_tone("beta1"));
        assert!(!f.has_pinyin_tone("babala2"));
        assert!(!f.has_pinyin_tone("voice2"));
    }

    #[test]
    fn correct_pinyin_maps_jqx() {
        assert_eq!(correct_pinyin("ju4"), "JV4");
        assert_eq!(correct_pinyin("que4"), "QVE4");
        assert_eq!(correct_pinyin("xun4"), "XVN4");
        assert_eq!(correct_pinyin("zhong4"), "zhong4");
    }

    #[test]
    fn chinese_cardinals() {
        assert_eq!(chinese_cardinal(0), "零");
        assert_eq!(chinese_cardinal(10), "十");
        assert_eq!(chinese_cardinal(15), "十五");
        assert_eq!(chinese_cardinal(20), "二十");
        assert_eq!(chinese_cardinal(105), "一百零五");
        assert_eq!(chinese_cardinal(1005), "一千零五");
        assert_eq!(chinese_cardinal(123), "一百二十三");
        assert_eq!(chinese_cardinal(2003), "二千零三");
        assert_eq!(chinese_cardinal(10000), "一万");
        assert_eq!(chinese_cardinal(100000), "十万");
        assert_eq!(chinese_cardinal(100030000), "一亿零三万");
    }

    #[test]
    fn zh_normalize_subset() {
        let f = TextFrontend::new();
        assert_eq!(f.normalize("共465篇", true), "共四百六十五篇");
        assert_eq!(f.normalize("效果666", true), "效果六百六十六");
        assert_eq!(f.normalize("2.5平方", true), "二点五平方");
        assert_eq!(f.normalize("2003年", true), "二零零三年");
        assert_eq!(f.normalize("折扣5%", true), "折扣百分之五");
    }

    #[test]
    fn english_cardinals_and_contractions() {
        assert_eq!(english_cardinal(2), "two");
        assert_eq!(english_cardinal(15), "fifteen");
        assert_eq!(english_cardinal(42), "forty-two");
        assert_eq!(english_cardinal(300), "three hundred");
        assert_eq!(
            english_cardinal(1234),
            "one thousand two hundred thirty-four"
        );
        assert_eq!(
            en_number_normalize("5G and 4G, 2.5%"),
            "fiveG and fourG, two point five%"
        );
        let f = TextFrontend::new();
        assert_eq!(
            f.normalize("where's the money?", false),
            "where is the money?"
        );
    }

    #[test]
    fn pronunciation_annotations() {
        assert_eq!(
            apply_pronunciation_annotations("<going|G OW1 . IH0 NG> ok"),
            "<|SPECIAL_TOKEN_1|>G OW1 . IH0 NG<|SPECIAL_TOKEN_1|> ok"
        );
        assert_eq!(
            apply_pronunciation_annotations("<行|XING2>"),
            "<|SPECIAL_TOKEN_2|>XING2<|SPECIAL_TOKEN_2|>"
        );
        assert_eq!(apply_pronunciation_annotations("plain"), "plain");
    }

    #[test]
    fn tech_terms_protect_hyphens() {
        let f = TextFrontend::new();
        // GPT-5 keeps its hyphen and the digit is still normalized.
        assert_eq!(f.normalize("GPT-5 发布", true), "GPT-五 发布");
    }
}
