//! misaki's English G2P (`misaki/en.py`, version 0.9.4), ported.
//!
//! Kokoro was trained on this module's output, quirks included, so the port
//! is deliberately literal: the same branches in the same order, the same
//! string predicates (Python's `isalpha`, `upper`, `capitalize`), and the
//! same odd corners (`is_currency` compares cents to the integer 0 and so
//! never matches; the `"None"` key in tagged entries). Where the Python
//! would raise, the port returns `None` and the word falls through to the
//! fallback. Function names follow the Python so the two can be read side
//! by side. misaki is Apache-2.0.

use std::collections::HashMap;
use std::sync::LazyLock;

use fancy_regex::Regex;
use unicode_normalization::UnicodeNormalization;

use crate::numbers;

// Every pattern below is a literal that compiles; a failure is a build
// defect, so `expect` is the honest response.
static SUBTOKEN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(concat!(
        r"^['‘’]+|\p{Lu}(?=\p{Lu}\p{Ll})|(?:^-)?(?:\d?[,.]?\d)+|[-_]+|['‘’]{2,}",
        r"|\p{L}*?(?:['‘’]\p{L})*?\p{Ll}(?=\p{Lu})|\p{L}+(?:['‘’]\p{L})*|[^-_\p{L}'‘’\d]|['‘’]+$",
    ))
    .expect("subtoken pattern")
});
static LINK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\[([^\]]+)\]\(([^\)]*)\)").expect("link pattern"));
static VS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?i)^vs\.?$").expect("vs pattern"));
static NUMBER_SUFFIX: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[a-z']+$").expect("suffix pattern"));
static NOT_LETTERS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[^a-z]+").expect("split pattern"));
static DOUBLED_ING: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"([bcdgklmnprstvxz])\1ing$|cking$").expect("ing pattern"));

const SUBTOKEN_JUNKS: &str = "',-._‘’/";
const PUNCTS: &str = ";:,.!?—…\"“”";
const NON_QUOTE_PUNCTS: &str = ";:,.!?—…";
const PUNCT_TAGS: [&str; 10] = [".", ",", "-LRB-", "-RRB-", "``", "\"\"", "''", ":", "$", "#"];
const CONSONANTS: &str = "bdfhjklmnpstvwzðŋɡɹɾʃʒʤʧθ";
const US_TAUS: &str = "AIOWYiuæɑəɛɪɹʊʌ";
const ORDINALS: [&str; 4] = ["st", "nd", "rd", "th"];
const DIPHTHONGS: &str = "AIOQWYʤʧ";
const VOWELS: &str = "AIOQWYaiuæɑɒɔəɛɜɪʊʌᵻ";
const PRIMARY: char = 'ˈ';
const SECONDARY: char = 'ˌ';
const CAP_STRESSES: (f64, f64) = (0.5, 2.0);

fn currency_units(symbol: &str) -> Option<(&'static str, &'static str)> {
    match symbol {
        "$" => Some(("dollar", "cent")),
        "£" => Some(("pound", "pence")),
        "€" => Some(("euro", "cent")),
        _ => None,
    }
}

fn punct_tag_phonemes(tag: &str) -> Option<&'static str> {
    match tag {
        "-LRB-" => Some("("),
        "-RRB-" => Some(")"),
        "``" => Some("\u{201c}"),
        "\"\"" | "''" => Some("\u{201d}"),
        _ => None,
    }
}

fn symbol_word(word: &str) -> Option<&'static str> {
    match word {
        "%" => Some("percent"),
        "&" => Some("and"),
        "+" => Some("plus"),
        "@" => Some("at"),
        _ => None,
    }
}

fn add_symbol_word(word: &str) -> Option<&'static str> {
    match word {
        "." => Some("dot"),
        "/" => Some("slash"),
        _ => None,
    }
}

// ---- Python string semantics -------------------------------------------

fn len(s: &str) -> usize {
    s.chars().count()
}

/// `s[:-n]`, by characters.
fn drop_last(s: &str, n: usize) -> &str {
    let keep = len(s).saturating_sub(n);
    s.char_indices().nth(keep).map_or(s, |(i, _)| &s[..i])
}

/// `s[1:]`, by characters.
fn tail(s: &str) -> &str {
    s.char_indices().nth(1).map_or("", |(i, _)| &s[i..])
}

fn last_char(s: &str) -> Option<char> {
    s.chars().next_back()
}

fn is_lower(s: &str) -> bool {
    s == s.to_lowercase()
}

fn is_upper(s: &str) -> bool {
    s == s.to_uppercase()
}

/// Python's `str.isalpha`: non-empty and every character a letter.
fn isalpha(s: &str) -> bool {
    !s.is_empty() && s.chars().all(char::is_alphabetic)
}

/// Python's `str.capitalize`: first character upper, the rest lower.
fn capitalize(s: &str) -> String {
    let mut chars = s.chars();
    chars.next().map_or_else(String::new, |f| f.to_uppercase().chain(chars.flat_map(char::to_lowercase)).collect())
}

/// misaki's `is_digit`: one or more ASCII digits.
fn is_digit(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit())
}

/// `ord(c) in LEXICON_ORDS`: an apostrophe, a hyphen or an ASCII letter.
fn is_lexicon_char(c: char) -> bool {
    c == '\'' || c == '-' || c.is_ascii_alphabetic()
}

// ---- Stress -------------------------------------------------------------

fn stress_weight(ps: &str) -> usize {
    ps.chars().map(|c| if DIPHTHONGS.contains(c) { 2 } else { 1 }).sum()
}

fn has_stress(ps: &str) -> bool {
    ps.contains(PRIMARY) || ps.contains(SECONDARY)
}

fn has_vowel(ps: &str) -> bool {
    ps.chars().any(|c| VOWELS.contains(c))
}

/// `restress(mark + ps)`: the mark moves to just before the first vowel.
fn stress_first_vowel(mark: char, ps: &str) -> String {
    let at = ps.char_indices().find(|&(_, c)| VOWELS.contains(c)).map_or(0, |(i, _)| i);
    format!("{}{mark}{}", &ps[..at], &ps[at..])
}

pub(crate) fn apply_stress(ps: &str, stress: Option<f64>) -> String {
    let Some(stress) = stress else { return ps.to_owned() };
    let strip = |ps: &str| ps.replace([PRIMARY, SECONDARY], "");
    if stress < -1.0 {
        strip(ps)
    } else if stress == -1.0 || ((stress == 0.0 || stress == -0.5) && ps.contains(PRIMARY)) {
        ps.replace(SECONDARY, "").replace(PRIMARY, &SECONDARY.to_string())
    } else if (stress == 0.0 || stress == 0.5 || stress == 1.0) && !has_stress(ps) {
        if has_vowel(ps) { stress_first_vowel(SECONDARY, ps) } else { ps.to_owned() }
    } else if stress >= 1.0 && !ps.contains(PRIMARY) && ps.contains(SECONDARY) {
        ps.replace(SECONDARY, &PRIMARY.to_string())
    } else if stress > 1.0 && !has_stress(ps) {
        if has_vowel(ps) { stress_first_vowel(PRIMARY, ps) } else { ps.to_owned() }
    } else {
        ps.to_owned()
    }
}

fn apply_stress_opt(ps: Option<String>, stress: Option<f64>) -> Option<String> {
    ps.map(|p| apply_stress(&p, stress))
}

// ---- Lexicon ------------------------------------------------------------

/// A lexicon value: one pronunciation, or one per part of speech (keys such
/// as `DEFAULT`, `VERB`, `NOUN`, `VBD`, and the literal string `"None"`).
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(untagged)]
pub(crate) enum Entry {
    One(String),
    Tagged(HashMap<String, Option<String>>),
}

impl Entry {
    fn default_str(&self) -> Option<&str> {
        match self {
            Entry::One(s) => Some(s),
            Entry::Tagged(m) => m.get("DEFAULT").and_then(|v| v.as_deref()),
        }
    }

    fn tagged(&self, tag: &str) -> Option<&str> {
        match self {
            Entry::Tagged(m) => m.get(tag).and_then(|v| v.as_deref()),
            Entry::One(_) => None,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TokenContext {
    pub future_vowel: Option<bool>,
    pub future_to: bool,
}

pub(crate) struct Lexicon {
    british: bool,
    golds: HashMap<String, Entry>,
    silvers: HashMap<String, Entry>,
}

impl Lexicon {
    pub fn new(british: bool, golds: HashMap<String, Entry>, silvers: HashMap<String, Entry>) -> Self {
        Self { british, golds: grow_dictionary(golds), silvers: grow_dictionary(silvers) }
    }

    fn gold_str(&self, word: &str) -> Option<String> {
        self.golds.get(word).and_then(Entry::default_str).map(str::to_owned)
    }

    /// Letter-by-letter reading, used by the spell-out fallback.
    pub fn spell(&self, word: &str) -> Option<String> {
        self.get_nnp(word)
    }

    /// The fallback for a word mixing letters, digits and separators
    /// ("9:30", "3:00", "config.toml"), which upstream hands to eSpeak NG
    /// whole. It reads the pieces the way eSpeak does: digit runs as numbers
    /// (a run with a leading zero digit by digit), letter runs as words,
    /// `:` and `.` kept as written, anything else dropped. A plain word goes
    /// straight to `word_fallback`.
    pub fn compose_fallback(&self, word: &str, word_fallback: &dyn Fn(&str) -> Option<String>) -> Option<String> {
        #[derive(PartialEq, Clone, Copy)]
        enum Kind {
            Letters,
            Digits,
            Other,
        }
        let kind = |c: char| {
            if c.is_alphabetic() || c == '\'' {
                Kind::Letters
            } else if c.is_ascii_digit() {
                Kind::Digits
            } else {
                Kind::Other
            }
        };
        if word.chars().all(|c| kind(c) == Kind::Letters || c == '-') {
            return word_fallback(word);
        }
        let mut runs: Vec<(Kind, String)> = Vec::new();
        for c in word.chars() {
            match runs.last_mut() {
                Some((k, run)) if *k == kind(c) && *k != Kind::Other => run.push(c),
                _ => runs.push((kind(c), c.to_string())),
            }
        }
        let mut out = String::new();
        for (k, run) in runs {
            let ps = match k {
                Kind::Letters => self.get_word(&run, "NN", None, TokenContext::default()).or_else(|| word_fallback(&run)),
                Kind::Digits => {
                    let mut words = Vec::new();
                    if run.len() > 1 && run.starts_with('0') {
                        for d in run.chars() {
                            self.extend_digits(&mut words, &d.to_string(), false, "")?;
                        }
                    } else {
                        self.extend_digits(&mut words, &run, true, "")?;
                    }
                    Some(words.join(" "))
                }
                Kind::Other if run == ":" || run == "." => Some(run),
                Kind::Other => None,
            };
            out.push_str(&ps.unwrap_or_default());
        }
        Some(out)
    }

    fn get_nnp(&self, word: &str) -> Option<String> {
        let mut ps = String::new();
        for c in word.chars().filter(|c| c.is_alphabetic()) {
            let upper: String = c.to_uppercase().collect();
            match self.golds.get(&upper) {
                Some(Entry::One(p)) => ps.push_str(p),
                _ => return None,
            }
        }
        let ps = apply_stress(&ps, Some(0.0));
        Some(match ps.rfind(SECONDARY) {
            Some(i) => format!("{}{PRIMARY}{}", &ps[..i], &ps[i + SECONDARY.len_utf8()..]),
            None => ps,
        })
    }

    fn get_special_case(&self, word: &str, tag: &str, stress: Option<f64>, ctx: TokenContext) -> Option<String> {
        if tag == "ADD"
            && let Some(w) = add_symbol_word(word)
        {
            return self.lookup(w, None, Some(-0.5), Some(ctx));
        }
        if let Some(w) = symbol_word(word) {
            return self.lookup(w, None, None, Some(ctx));
        }
        let stripped = word.trim_matches('.');
        if stripped.contains('.') && isalpha(&word.replace('.', "")) && word.split('.').map(len).max().unwrap_or(0) < 3 {
            return self.get_nnp(word);
        }
        match word {
            "a" | "A" => return Some(if tag == "DT" { "ɐ" } else { "ˈA" }.to_owned()),
            "am" | "Am" | "AM" => {
                if tag.starts_with("NN") {
                    return self.get_nnp(word);
                } else if ctx.future_vowel.is_none() || word != "am" || stress.is_some_and(|s| s > 0.0) {
                    return self.gold_str("am");
                }
                return Some("ɐm".to_owned());
            }
            "an" | "An" | "AN" => {
                if word == "AN" && tag.starts_with("NN") {
                    return self.get_nnp(word);
                }
                return Some("ɐn".to_owned());
            }
            "I" if tag == "PRP" => return Some(format!("{SECONDARY}I")),
            "by" | "By" | "BY" if parent_tag(Some(tag)) == Some("ADV") => return Some("bˈI".to_owned()),
            _ => {}
        }
        if matches!(word, "to" | "To") || (word == "TO" && matches!(tag, "TO" | "IN")) {
            return match ctx.future_vowel {
                None => self.gold_str("to"),
                Some(false) => Some("tə".to_owned()),
                Some(true) => Some("tʊ".to_owned()),
            };
        }
        if matches!(word, "in" | "In") || (word == "IN" && tag != "NNP") {
            let stress = if ctx.future_vowel.is_none() || tag != "IN" { "ˈ" } else { "" };
            return Some(format!("{stress}ɪn"));
        }
        if matches!(word, "the" | "The") || (word == "THE" && tag == "DT") {
            return Some(if ctx.future_vowel == Some(true) { "ði" } else { "ðə" }.to_owned());
        }
        if tag == "IN" && VS.is_match(word).unwrap_or(false) {
            return self.lookup("versus", None, None, Some(ctx));
        }
        if matches!(word, "used" | "Used" | "USED") {
            let used = self.golds.get("used")?;
            let key = if matches!(tag, "VBD" | "JJ") && ctx.future_to { "VBD" } else { "DEFAULT" };
            return used.tagged(key).map(str::to_owned);
        }
        None
    }

    fn is_known(&self, word: &str) -> bool {
        if self.golds.contains_key(word) || symbol_word(word).is_some() || self.silvers.contains_key(word) {
            return true;
        }
        if !isalpha(word) || !word.chars().all(is_lexicon_char) {
            return false;
        }
        if len(word) == 1 {
            return true;
        }
        if is_upper(word) && self.golds.contains_key(&word.to_lowercase()) {
            return true;
        }
        is_upper(tail(word))
    }

    fn lookup(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<TokenContext>) -> Option<String> {
        let mut word = word.to_owned();
        let mut is_nnp = false;
        if is_upper(&word) && !self.golds.contains_key(&word) {
            word = word.to_lowercase();
            is_nnp = tag == Some("NNP");
        }
        let mut entry = self.golds.get(&word);
        if entry.is_none() && !is_nnp {
            entry = self.silvers.get(&word);
        }
        let ps: Option<String> = match entry {
            None => None,
            Some(Entry::One(p)) => Some(p.clone()),
            Some(Entry::Tagged(m)) => {
                let key: Option<String> = if ctx.is_some_and(|c| c.future_vowel.is_none()) && m.contains_key("None") {
                    Some("None".to_owned())
                } else if tag.is_some_and(|t| m.contains_key(t)) {
                    tag.map(str::to_owned)
                } else {
                    parent_tag(tag).map(str::to_owned)
                };
                // `ps.get(tag, ps['DEFAULT'])`: a present key wins even
                // when its value is null.
                match key.as_deref().and_then(|k| m.get(k)) {
                    Some(v) => v.clone(),
                    None => m.get("DEFAULT").cloned().flatten(),
                }
            }
        };
        if (ps.is_none() || (is_nnp && !ps.as_deref().is_some_and(|p| p.contains(PRIMARY))))
            && let Some(nnp) = self.get_nnp(&word)
        {
            return Some(nnp);
        }
        apply_stress_opt(ps, stress)
    }

    fn suffix_s(&self, stem: Option<String>) -> Option<String> {
        let stem = stem.filter(|s| !s.is_empty())?;
        let last = last_char(&stem)?;
        Some(if "ptkfθ".contains(last) {
            stem + "s"
        } else if "szʃʒʧʤ".contains(last) {
            stem + if self.british { "ɪz" } else { "ᵻz" }
        } else {
            stem + "z"
        })
    }

    fn stem_s(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<TokenContext>) -> Option<String> {
        if len(word) < 3 || !word.ends_with('s') {
            return None;
        }
        let stem = if !word.ends_with("ss") && self.is_known(drop_last(word, 1)) {
            drop_last(word, 1).to_owned()
        } else if (word.ends_with("'s") || (len(word) > 4 && word.ends_with("es") && !word.ends_with("ies")))
            && self.is_known(drop_last(word, 2))
        {
            drop_last(word, 2).to_owned()
        } else if len(word) > 4 && word.ends_with("ies") && self.is_known(&format!("{}y", drop_last(word, 3))) {
            format!("{}y", drop_last(word, 3))
        } else {
            return None;
        };
        self.suffix_s(self.lookup(&stem, tag, stress, ctx))
    }

    fn suffix_ed(&self, stem: Option<String>) -> Option<String> {
        let stem = stem.filter(|s| !s.is_empty())?;
        let last = last_char(&stem)?;
        Some(if "pkfθʃsʧ".contains(last) {
            stem + "t"
        } else if last == 'd' {
            stem + if self.british { "ɪd" } else { "ᵻd" }
        } else if last != 't' {
            stem + "d"
        } else if self.british || len(&stem) < 2 {
            stem + "ɪd"
        } else if stem.chars().rev().nth(1).is_some_and(|c| US_TAUS.contains(c)) {
            format!("{}ɾᵻd", drop_last(&stem, 1))
        } else {
            stem + "ᵻd"
        })
    }

    fn stem_ed(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<TokenContext>) -> Option<String> {
        if len(word) < 4 || !word.ends_with('d') {
            return None;
        }
        let stem = if !word.ends_with("dd") && self.is_known(drop_last(word, 1)) {
            drop_last(word, 1)
        } else if len(word) > 4 && word.ends_with("ed") && !word.ends_with("eed") && self.is_known(drop_last(word, 2)) {
            drop_last(word, 2)
        } else {
            return None;
        };
        self.suffix_ed(self.lookup(stem, tag, stress, ctx))
    }

    fn suffix_ing(&self, stem: Option<String>) -> Option<String> {
        let stem = stem.filter(|s| !s.is_empty())?;
        let last = last_char(&stem)?;
        if self.british {
            if "əː".contains(last) {
                return None;
            }
        } else if len(&stem) > 1 && last == 't' && stem.chars().rev().nth(1).is_some_and(|c| US_TAUS.contains(c)) {
            return Some(format!("{}ɾɪŋ", drop_last(&stem, 1)));
        }
        Some(stem + "ɪŋ")
    }

    fn stem_ing(&self, word: &str, tag: Option<&str>, stress: Option<f64>, ctx: Option<TokenContext>) -> Option<String> {
        if len(word) < 5 || !word.ends_with("ing") {
            return None;
        }
        let stem = if len(word) > 5 && self.is_known(drop_last(word, 3)) {
            drop_last(word, 3).to_owned()
        } else if self.is_known(&format!("{}e", drop_last(word, 3))) {
            format!("{}e", drop_last(word, 3))
        } else if len(word) > 5 && DOUBLED_ING.is_match(word).unwrap_or(false) && self.is_known(drop_last(word, 4)) {
            drop_last(word, 4).to_owned()
        } else {
            return None;
        };
        self.suffix_ing(self.lookup(&stem, tag, stress, ctx))
    }

    fn get_word(&self, word: &str, tag: &str, stress: Option<f64>, ctx: TokenContext) -> Option<String> {
        if let Some(ps) = self.get_special_case(word, tag, stress, ctx) {
            return Some(ps);
        }
        let wl = word.to_lowercase();
        let tag_opt = Some(tag);
        let mut word = word.to_owned();
        if len(&word) > 1
            && isalpha(&word.replace('\'', ""))
            && word != wl
            && (tag != "NNP" || len(&word) > 7)
            && !self.golds.contains_key(&word)
            && !self.silvers.contains_key(&word)
            && (is_upper(&word) || is_lower(tail(&word)))
            && (self.golds.contains_key(&wl)
                || self.silvers.contains_key(&wl)
                || self.stem_s(&wl, tag_opt, stress, Some(ctx)).is_some()
                || self.stem_ed(&wl, tag_opt, stress, Some(ctx)).is_some()
                || self.stem_ing(&wl, tag_opt, stress, Some(ctx)).is_some())
        {
            word = wl;
        }
        if self.is_known(&word) {
            return self.lookup(&word, tag_opt, stress, Some(ctx));
        }
        if word.ends_with("s'") {
            let possessive = format!("{}'s", drop_last(&word, 2));
            if self.is_known(&possessive) {
                return self.lookup(&possessive, tag_opt, stress, Some(ctx));
            }
        }
        if word.ends_with('\'') && self.is_known(drop_last(&word, 1)) {
            return self.lookup(drop_last(&word, 1), tag_opt, stress, Some(ctx));
        }
        if let Some(s) = self.stem_s(&word, tag_opt, stress, Some(ctx)) {
            return Some(s);
        }
        if let Some(ed) = self.stem_ed(&word, tag_opt, stress, Some(ctx)) {
            return Some(ed);
        }
        self.stem_ing(&word, tag_opt, Some(stress.unwrap_or(0.5)), Some(ctx))
    }

    fn get_number(&self, word: &str, currency: Option<&str>, is_head: bool, num_flags: &str) -> Option<String> {
        let suffix = NUMBER_SUFFIX.find(word).ok().flatten().map(|m| m.as_str().to_owned());
        let mut word = match &suffix {
            Some(s) => drop_last(word, len(s)).to_owned(),
            None => word.to_owned(),
        };
        let suffix = suffix.as_deref();
        let mut result: Vec<String> = Vec::new();
        if let Some(rest) = word.strip_prefix('-') {
            result.push(self.lookup("minus", None, None, None)?);
            word = rest.to_owned();
        }
        let is_ordinal = suffix.is_some_and(|s| ORDINALS.contains(&s));
        let currency_units = currency.and_then(currency_units);

        if is_digit(&word) && is_ordinal {
            self.extend_words(&mut result, &numbers::ordinal(word.parse().ok()?), true, num_flags)?;
        } else if result.is_empty() && len(&word) == 4 && currency_units.is_none() && is_digit(&word) {
            self.extend_words(&mut result, &numbers::year(word.parse().ok()?), true, num_flags)?;
        } else if !is_head && !word.contains('.') {
            let num = word.replace(',', "");
            if num.starts_with('0') || len(&num) > 3 {
                for d in num.chars() {
                    self.extend_digits(&mut result, &d.to_string(), false, num_flags)?;
                }
            } else if len(&num) == 3 && !num.ends_with("00") {
                self.extend_digits(&mut result, &num[..1], true, num_flags)?;
                if &num[1..2] == "0" {
                    result.push(self.lookup("O", None, Some(-2.0), None)?);
                    self.extend_digits(&mut result, &num[2..], false, num_flags)?;
                } else {
                    self.extend_digits(&mut result, &num[1..], false, num_flags)?;
                }
            } else {
                self.extend_digits(&mut result, &num, true, num_flags)?;
            }
        } else if word.matches('.').count() > 1 || !is_head {
            let mut first = true;
            for num in word.replace(',', "").split('.') {
                if num.is_empty() {
                } else if num.starts_with('0') || (len(num) != 2 && num.chars().skip(1).any(|c| c != '0')) {
                    for d in num.chars() {
                        self.extend_digits(&mut result, &d.to_string(), false, num_flags)?;
                    }
                } else {
                    self.extend_digits(&mut result, num, first, num_flags)?;
                }
                first = false;
            }
        } else if let (Some((major, minor)), true) = (currency_units, is_currency(&word)) {
            let plain = word.replace(',', "");
            let mut pairs: Vec<(u64, &str)> = plain
                .split('.')
                .zip([major, minor])
                .map(|(n, unit)| Some((if n.is_empty() { 0 } else { n.parse().ok()? }, unit)))
                .collect::<Option<_>>()?;
            if pairs.len() > 1 {
                if pairs[1].0 == 0 {
                    pairs.truncate(1);
                } else if pairs[0].0 == 0 {
                    pairs.remove(0);
                }
            }
            for (i, (num, unit)) in pairs.into_iter().enumerate() {
                if i > 0 {
                    result.push(self.lookup("and", None, None, None)?);
                }
                self.extend_digits(&mut result, &num.to_string(), i == 0, num_flags)?;
                let unit_ps = if num != 1 && unit != "pence" {
                    self.stem_s(&format!("{unit}s"), None, None, None)
                } else {
                    self.lookup(unit, None, None, None)
                };
                result.push(unit_ps?);
            }
        } else {
            let words = if is_digit(&word) {
                numbers::cardinal(word.parse().ok()?)
            } else if !word.contains('.') {
                let n: u64 = word.replace(',', "").parse().ok()?;
                if is_ordinal { numbers::ordinal(n) } else { numbers::cardinal(n) }
            } else {
                let plain = word.replace(',', "");
                if let Some(frac) = plain.strip_prefix('.') {
                    let digits: Vec<String> =
                        frac.chars().map(|d| d.to_digit(10).map(|d| numbers::cardinal(d.into()))).collect::<Option<_>>()?;
                    format!("point {}", digits.join(" "))
                } else {
                    numbers::decimal(&plain)?
                }
            };
            self.extend_words(&mut result, &words, true, num_flags)?;
        }
        if result.is_empty() {
            return None;
        }
        let joined = result.join(" ");
        match suffix {
            Some("s" | "'s") => self.suffix_s(Some(joined)),
            Some("ed" | "'d") => self.suffix_ed(Some(joined)),
            Some("ing") => self.suffix_ing(Some(joined)),
            _ => Some(joined),
        }
    }

    /// `extend_num(num)` with `escape=False`: spell the digits, then add.
    fn extend_digits(&self, result: &mut Vec<String>, digits: &str, first: bool, num_flags: &str) -> Option<()> {
        self.extend_words(result, &numbers::cardinal(digits.parse().ok()?), first, num_flags)
    }

    /// `extend_num(words, escape=True)`: phonemize each word, dropping "and"
    /// unless the `&` flag keeps it.
    fn extend_words(&self, result: &mut Vec<String>, words: &str, first: bool, num_flags: &str) -> Option<()> {
        let splits: Vec<&str> = NOT_LETTERS.split(words).filter_map(Result::ok).collect();
        for (i, w) in splits.iter().enumerate() {
            if *w != "and" || num_flags.contains('&') {
                if first && i == 0 && splits.len() > 1 && *w == "one" && num_flags.contains('a') {
                    result.push("ə".to_owned());
                } else {
                    let stress = if *w == "point" { Some(-2.0) } else { None };
                    result.push(self.lookup(w, None, stress, None)?);
                }
            } else if num_flags.contains('n')
                && let Some(last) = result.last_mut()
            {
                last.push_str("ən");
            }
        }
        Some(())
    }

    fn append_currency(&self, ps: String, currency: Option<&str>) -> String {
        let Some(currency) = currency.filter(|c| !c.is_empty()) else { return ps };
        match currency_units(currency).and_then(|(major, _)| self.stem_s(&format!("{major}s"), None, None, None)) {
            Some(unit) => format!("{ps} {unit}"),
            None => ps,
        }
    }

    /// `Lexicon.__call__`: phonemes for one token, or `None` for the
    /// fallback to handle.
    pub fn phonemize(&self, tk: &MToken, ctx: TokenContext) -> Option<String> {
        let source = tk.alias.as_deref().unwrap_or(&tk.text).replace(['‘', '’'], "'");
        let word: String = source.nfkc().map(numeric_if_needed).collect();
        let stress = if is_lower(&word) {
            None
        } else if is_upper(&word) {
            Some(CAP_STRESSES.1)
        } else {
            Some(CAP_STRESSES.0)
        };
        if let Some(ps) = self.get_word(&word, &tk.tag, stress, ctx) {
            return Some(apply_stress(&self.append_currency(ps, tk.currency.as_deref()), tk.stress));
        }
        if is_number(&word, tk.is_head) {
            let ps = self.get_number(&word, tk.currency.as_deref(), tk.is_head, &tk.num_flags)?;
            return Some(apply_stress(&ps, tk.stress));
        }
        None
    }
}

fn parent_tag(tag: Option<&str>) -> Option<&str> {
    let t = tag?;
    Some(if t.starts_with("VB") {
        "VERB"
    } else if t.starts_with("NN") {
        "NOUN"
    } else if t.starts_with("ADV") || t.starts_with("RB") {
        "ADV"
    } else if t.starts_with("ADJ") || t.starts_with("JJ") {
        "ADJ"
    } else {
        t
    })
}

/// Adds the capitalised form of each lowercase key and the lowercase form
/// of each capitalised key; entries already present win.
fn grow_dictionary(d: HashMap<String, Entry>) -> HashMap<String, Entry> {
    let mut grown: HashMap<String, Entry> = HashMap::with_capacity(d.len() * 2);
    for (k, v) in &d {
        if len(k) < 2 {
            continue;
        }
        if is_lower(k) {
            let cap = capitalize(k);
            if *k != cap {
                grown.insert(cap, v.clone());
            }
        } else if *k == capitalize(&k.to_lowercase()) {
            grown.insert(k.to_lowercase(), v.clone());
        }
    }
    grown.extend(d);
    grown
}

/// `Lexicon.is_currency`. The Python compares the cents' characters to the
/// integer 0, which never matches, so only the length test survives.
fn is_currency(word: &str) -> bool {
    match word.matches('.').count() {
        0 => true,
        1 => word.split('.').nth(1).is_some_and(|cents| len(cents) < 3),
        _ => false,
    }
}

/// `Lexicon.numeric_if_needed`: a digit NFKC left alone (Arabic-Indic,
/// Devanagari, ...) becomes its ASCII value.
fn numeric_if_needed(c: char) -> char {
    if c.is_ascii_digit() || !c.is_numeric() {
        return c;
    }
    // Unicode decimal digits come in contiguous 0-9 runs.
    const ZEROS: [u32; 12] = [0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6, 0xC66, 0xCE6, 0xD66];
    let code = c as u32;
    ZEROS.iter().find(|&&z| (z..z + 10).contains(&code)).and_then(|z| char::from_digit(code - z, 10)).unwrap_or(c)
}

fn is_number(word: &str, is_head: bool) -> bool {
    if !word.chars().any(|c| c.is_ascii_digit()) {
        return false;
    }
    let mut word = word;
    for s in ["ing", "'d", "ed", "'s", "st", "nd", "rd", "th", "s"] {
        if let Some(stripped) = word.strip_suffix(s) {
            word = stripped;
            break;
        }
    }
    word.chars().enumerate().all(|(i, c)| c.is_ascii_digit() || c == ',' || c == '.' || (is_head && i == 0 && c == '-'))
}

// ---- The pipeline -------------------------------------------------------

/// misaki's `MToken` with its `_` extension fields flattened in.
#[derive(Debug, Clone, Default)]
pub(crate) struct MToken {
    pub text: String,
    pub tag: String,
    pub whitespace: String,
    pub phonemes: Option<String>,
    pub is_head: bool,
    pub alias: Option<String>,
    pub stress: Option<f64>,
    pub currency: Option<String>,
    pub num_flags: String,
    pub prespace: bool,
}

/// A word after retokenization: one token, or several written together
/// ("don't" as "do" + "n't", "B22" as "B" + "22").
enum Word {
    One(MToken),
    Many(Vec<MToken>),
}

/// A `[text](feature)` annotation from the input, per misaki's markup.
#[derive(Debug, Clone)]
enum Feature {
    Stress(f64),
    Phonemes(String),
    NumFlags(String),
}

pub(crate) fn merge_tokens(tokens: &[MToken], unk: Option<&str>) -> MToken {
    let stresses: Vec<f64> = tokens.iter().filter_map(|t| t.stress).collect();
    let stress = match stresses.first() {
        Some(&s) if stresses.iter().all(|&x| x == s) => Some(s),
        _ => None,
    };
    let currency = tokens.iter().filter_map(|t| t.currency.clone()).max();
    let phonemes = unk.map(|unk| {
        let mut ps = String::new();
        for tk in tokens {
            if tk.prespace
                && !ps.is_empty()
                && !last_char(&ps).is_some_and(char::is_whitespace)
                && tk.phonemes.as_deref().is_some_and(|p| !p.is_empty())
            {
                ps.push(' ');
            }
            ps.push_str(tk.phonemes.as_deref().unwrap_or(unk));
        }
        ps
    });
    let mut text = String::new();
    for (i, tk) in tokens.iter().enumerate() {
        text.push_str(&tk.text);
        if i + 1 < tokens.len() {
            text.push_str(&tk.whitespace);
        }
    }
    // The tag of the most "capitalised" token, first among equals.
    let weight = |t: &MToken| t.text.chars().map(|c| if c.to_lowercase().eq(std::iter::once(c)) { 1 } else { 2 }).sum::<usize>();
    let tag = tokens.iter().fold(None::<&MToken>, |best, t| match best {
        Some(b) if weight(b) >= weight(t) => Some(b),
        _ => Some(t),
    });
    let mut num_flags: Vec<char> = tokens.iter().flat_map(|t| t.num_flags.chars()).collect();
    num_flags.sort_unstable();
    num_flags.dedup();
    MToken {
        text,
        tag: tag.map(|t| t.tag.clone()).unwrap_or_default(),
        whitespace: tokens.last().map(|t| t.whitespace.clone()).unwrap_or_default(),
        phonemes,
        is_head: tokens.first().is_some_and(|t| t.is_head),
        alias: None,
        stress,
        currency,
        num_flags: num_flags.into_iter().collect(),
        prespace: tokens.first().is_some_and(|t| t.prespace),
    }
}

/// `G2P.preprocess`: strips `[text](feature)` markup, returning the plain
/// text, its whitespace-separated words, and features keyed by word index.
fn preprocess(text: &str) -> (String, Vec<String>, HashMap<usize, Feature>) {
    let text = text.trim_start();
    let mut result = String::new();
    let mut words: Vec<String> = Vec::new();
    let mut features = HashMap::new();
    let mut last_end = 0;
    for caps in LINK.captures_iter(text).filter_map(Result::ok) {
        let whole = caps.get(0).expect("group 0");
        let (inner, f) = (caps.get(1).map_or("", |m| m.as_str()), caps.get(2).map_or("", |m| m.as_str()));
        result.push_str(&text[last_end..whole.start()]);
        words.extend(text[last_end..whole.start()].split_whitespace().map(str::to_owned));
        let signless = f.strip_prefix(['-', '+']).unwrap_or(f);
        let feature = if is_digit(signless) {
            f.parse::<i64>().ok().map(|n| Feature::Stress(n as f64))
        } else if f == "0.5" || f == "+0.5" {
            Some(Feature::Stress(0.5))
        } else if f == "-0.5" {
            Some(Feature::Stress(-0.5))
        } else if len(f) > 1 && f.starts_with('/') && f.ends_with('/') {
            Some(Feature::Phonemes(f[1..].trim_end_matches('/').to_owned()))
        } else if len(f) > 1 && f.starts_with('#') && f.ends_with('#') {
            Some(Feature::NumFlags(f[1..].trim_end_matches('#').to_owned()))
        } else {
            None
        };
        if let Some(feature) = feature {
            features.insert(words.len(), feature);
        }
        result.push_str(inner);
        words.push(inner.to_owned());
        last_end = whole.end();
    }
    if last_end < text.len() {
        result.push_str(&text[last_end..]);
        words.extend(text[last_end..].split_whitespace().map(str::to_owned));
    }
    (result, words, features)
}

/// Maps each token to the index of the whitespace-separated word it falls
/// in, comparing the two segmentations with whitespace removed (what
/// spaCy's `Alignment.from_strings` does).
fn align(words: &[String], tokens: &[MToken]) -> Vec<Option<usize>> {
    let mut owner = Vec::new();
    for (w, word) in words.iter().enumerate() {
        owner.extend(word.chars().filter(|c| !c.is_whitespace()).map(|_| w));
    }
    let mut at = 0;
    tokens
        .iter()
        .map(|t| {
            let n = t.text.chars().filter(|c| !c.is_whitespace()).count();
            let w = if n == 0 { None } else { owner.get(at).copied() };
            at += n;
            w
        })
        .collect()
}

fn apply_features(tokens: &mut [MToken], words: &[String], features: &HashMap<usize, Feature>) {
    if features.is_empty() {
        return;
    }
    let owners = align(words, tokens);
    for (&k, feature) in features {
        let aligned = owners.iter().enumerate().filter(|(_, o)| **o == Some(k)).map(|(j, _)| j);
        for (i, j) in aligned.enumerate() {
            let tk = &mut tokens[j];
            match feature {
                Feature::Stress(s) => tk.stress = Some(*s),
                Feature::Phonemes(ps) => {
                    tk.is_head = i == 0;
                    tk.phonemes = Some(if i == 0 { ps.clone() } else { String::new() });
                }
                Feature::NumFlags(flags) => tk.num_flags = flags.clone(),
            }
        }
    }
}

fn fold_left(tokens: Vec<MToken>, unk: &str) -> Vec<MToken> {
    let mut result: Vec<MToken> = Vec::with_capacity(tokens.len());
    for tk in tokens {
        match result.pop() {
            Some(prev) if !tk.is_head => result.push(merge_tokens(&[prev, tk], Some(unk))),
            Some(prev) => {
                result.push(prev);
                result.push(tk);
            }
            None => result.push(tk),
        }
    }
    result
}

fn subtokenize(text: &str) -> Vec<&str> {
    SUBTOKEN.find_iter(text).filter_map(Result::ok).map(|m| m.as_str()).collect()
}

fn retokenize(tokens: Vec<MToken>) -> Vec<Word> {
    let tags: Vec<String> = tokens.iter().map(|t| t.tag.clone()).collect();
    let mut words: Vec<Word> = Vec::new();
    let mut currency: Option<String> = None;
    let count = tokens.len();
    for (i, token) in tokens.into_iter().enumerate() {
        let mut tks: Vec<MToken> = if token.alias.is_none() && token.phonemes.is_none() {
            subtokenize(&token.text)
                .into_iter()
                .map(|t| MToken {
                    text: t.to_owned(),
                    tag: token.tag.clone(),
                    whitespace: String::new(),
                    is_head: true,
                    num_flags: token.num_flags.clone(),
                    stress: token.stress,
                    ..MToken::default()
                })
                .collect()
        } else {
            vec![token.clone()]
        };
        if let Some(last) = tks.last_mut() {
            last.whitespace = token.whitespace.clone();
        }
        let n = tks.len();
        let texts: Vec<String> = tks.iter().map(|t| t.text.clone()).collect();
        for (j, mut tk) in tks.into_iter().enumerate() {
            if tk.alias.is_some() || tk.phonemes.is_some() {
            } else if tk.tag == "$" && currency_units(&tk.text).is_some() {
                currency = Some(tk.text.clone());
                tk.phonemes = Some(String::new());
            } else if tk.tag == ":" && (tk.text == "-" || tk.text == "–") {
                tk.phonemes = Some("—".to_owned());
            } else if PUNCT_TAGS.contains(&tk.tag.as_str()) && !tk.text.chars().all(|c| c.is_ascii_alphabetic()) {
                tk.phonemes = Some(match punct_tag_phonemes(&tk.tag) {
                    Some(p) => p.to_owned(),
                    None => tk.text.chars().filter(|c| PUNCTS.contains(*c)).collect(),
                });
            } else if currency.is_some() {
                if tk.tag != "CD" {
                    currency = None;
                } else if j + 1 == n && (i + 1 == count || tags[i + 1] != "CD") {
                    tk.currency = currency.clone();
                }
            } else if 0 < j && j + 1 < n && tk.text == "2" {
                let around: String = last_char(&texts[j - 1]).into_iter().chain(texts[j + 1].chars().next()).collect();
                if isalpha(&around) {
                    tk.alias = Some("to".to_owned());
                }
            }

            if tk.alias.is_some() || tk.phonemes.is_some() {
                words.push(Word::One(tk));
            } else if let Some(Word::Many(group)) =
                words.last_mut().filter(|w| matches!(w, Word::Many(g) if g.last().is_some_and(|t| t.whitespace.is_empty())))
            {
                tk.is_head = false;
                group.push(tk);
            } else if tk.whitespace.is_empty() {
                words.push(Word::Many(vec![tk]));
            } else {
                words.push(Word::One(tk));
            }
        }
    }
    words
        .into_iter()
        .map(|w| match w {
            Word::Many(mut g) if g.len() == 1 => Word::One(g.remove(0)),
            other => other,
        })
        .collect()
}

fn token_context(ctx: TokenContext, ps: Option<&str>, token: &MToken) -> TokenContext {
    let mut vowel = ctx.future_vowel;
    if let Some(ps) = ps.filter(|p| !p.is_empty()) {
        let found = ps.chars().find(|&c| VOWELS.contains(c) || CONSONANTS.contains(c) || NON_QUOTE_PUNCTS.contains(c));
        if let Some(c) = found {
            vowel = if NON_QUOTE_PUNCTS.contains(c) { None } else { Some(VOWELS.contains(c)) };
        }
    }
    let future_to = matches!(token.text.as_str(), "to" | "To") || (token.text == "TO" && matches!(token.tag.as_str(), "TO" | "IN"));
    TokenContext { future_vowel: vowel, future_to }
}

fn resolve_tokens(tokens: &mut [MToken]) {
    let mut text = String::new();
    for (i, tk) in tokens.iter().enumerate() {
        text.push_str(&tk.text);
        if i + 1 < tokens.len() {
            text.push_str(&tk.whitespace);
        }
    }
    let kinds: std::collections::HashSet<u8> = text
        .chars()
        .filter(|c| !SUBTOKEN_JUNKS.contains(*c))
        .map(|c| {
            if c.is_alphabetic() {
                0
            } else if c.is_ascii_digit() {
                1
            } else {
                2
            }
        })
        .collect();
    let prespace = text.contains(' ') || text.contains('/') || kinds.len() > 1;
    let last = tokens.len() - 1;
    for (i, tk) in tokens.iter_mut().enumerate() {
        if tk.phonemes.is_none() {
            if i == last && tk.text.chars().count() == 1 && NON_QUOTE_PUNCTS.contains(tk.text.as_str()) {
                tk.phonemes = Some(tk.text.clone());
            } else if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                tk.phonemes = Some(String::new());
            }
        } else if i > 0 {
            tk.prespace = prespace;
        }
    }
    if prespace {
        return;
    }
    let mut indices: Vec<(bool, usize, usize)> = tokens
        .iter()
        .enumerate()
        .filter_map(|(i, tk)| tk.phonemes.as_deref().filter(|p| !p.is_empty()).map(|p| (p.contains(PRIMARY), stress_weight(p), i)))
        .collect();
    if indices.len() == 2 && tokens[indices[0].2].text.chars().count() == 1 {
        let i = indices[1].2;
        tokens[i].phonemes = apply_stress_opt(tokens[i].phonemes.take(), Some(-0.5));
        return;
    }
    let primaries = indices.iter().filter(|(b, _, _)| *b).count();
    if indices.len() < 2 || primaries <= indices.len().div_ceil(2) {
        return;
    }
    indices.sort();
    for &(_, _, i) in &indices[..indices.len() / 2] {
        tokens[i].phonemes = apply_stress_opt(tokens[i].phonemes.take(), Some(-0.5));
    }
}

/// Runs the whole pipeline over pre-tagged tokens and returns one token per
/// spoken word, phonemes filled in (`None` where nothing could be found).
/// `fallback` gets the text of any word the lexicon cannot handle; `unk`
/// stands in for an unphonemizable piece inside a multi-part word.
pub(crate) fn g2p(
    lexicon: &Lexicon,
    text: &str,
    tokenize: impl FnOnce(&str) -> Vec<MToken>,
    fallback: &dyn Fn(&str) -> Option<String>,
    unk: &str,
) -> Vec<MToken> {
    let (text, words, features) = preprocess(text);
    let mut tokens = tokenize(&text);
    apply_features(&mut tokens, &words, &features);
    let tokens = fold_left(tokens, unk);
    let mut words = retokenize(tokens);

    let mut ctx = TokenContext::default();
    for w in words.iter_mut().rev() {
        match w {
            Word::One(tk) => {
                if tk.phonemes.is_none() {
                    tk.phonemes = lexicon.phonemize(tk, ctx);
                }
                if tk.phonemes.is_none() {
                    tk.phonemes = fallback(&tk.text);
                }
                ctx = token_context(ctx, tk.phonemes.as_deref(), tk);
            }
            Word::Many(group) => {
                let (mut left, mut right) = (0, group.len());
                let mut should_fallback = false;
                while left < right {
                    let span = &group[left..right];
                    let merged =
                        if span.iter().any(|t| t.alias.is_some() || t.phonemes.is_some()) { None } else { Some(merge_tokens(span, None)) };
                    let ps = merged.as_ref().and_then(|tk| lexicon.phonemize(tk, ctx));
                    if let (Some(ps), Some(tk)) = (ps, merged.as_ref()) {
                        ctx = token_context(ctx, Some(&ps), tk);
                        group[left].phonemes = Some(ps);
                        for x in &mut group[left + 1..right] {
                            x.phonemes = Some(String::new());
                        }
                        right = left;
                        left = 0;
                    } else if left + 1 < right {
                        left += 1;
                    } else {
                        right -= 1;
                        let tk = &mut group[right];
                        if tk.phonemes.is_none() {
                            if tk.text.chars().all(|c| SUBTOKEN_JUNKS.contains(c)) {
                                tk.phonemes = Some(String::new());
                            } else {
                                should_fallback = true;
                                break;
                            }
                        }
                        left = 0;
                    }
                }
                if should_fallback {
                    let merged = merge_tokens(group, None);
                    group[0].phonemes = fallback(&merged.text);
                    for x in &mut group[1..] {
                        x.phonemes = Some(String::new());
                    }
                } else {
                    resolve_tokens(group);
                }
            }
        }
    }

    words
        .into_iter()
        .map(|w| {
            let mut tk = match w {
                Word::One(tk) => tk,
                Word::Many(group) => merge_tokens(&group, Some(unk)),
            };
            // misaki's output notation: the flap as T, the glottal stop as t.
            tk.phonemes = tk.phonemes.map(|ps| ps.replace('ɾ', "T").replace('ʔ', "t"));
            tk
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stress_rules_match_misaki() {
        assert_eq!(apply_stress("həlˈO", Some(-2.0)), "həlO");
        assert_eq!(apply_stress("həlˈO", Some(-1.0)), "həlˌO");
        assert_eq!(apply_stress("wʌn", Some(0.0)), "wˌʌn");
        assert_eq!(apply_stress("wʌn", Some(2.0)), "wˈʌn");
        assert_eq!(apply_stress("ˌaɪ", Some(1.0)), "ˈaɪ");
        assert_eq!(apply_stress("bd", Some(2.0)), "bd");
    }

    #[test]
    fn python_string_helpers() {
        assert_eq!(drop_last("café", 1), "caf");
        assert_eq!(tail("Word"), "ord");
        assert_eq!(capitalize("hELLO"), "Hello");
        assert!(isalpha("Straße") && !isalpha("don't") && !isalpha(""));
    }

    #[test]
    fn currency_test_keeps_the_upstream_quirk() {
        assert!(is_currency("12"));
        assert!(is_currency("12.50"));
        assert!(!is_currency("12.500"));
        assert!(!is_currency("1.2.3"));
    }

    #[test]
    fn number_detection() {
        assert!(is_number("3rd", true));
        assert!(is_number("-5", true));
        assert!(!is_number("-5", false));
        assert!(is_number("1,234.5", true));
        assert!(!is_number("B22", true));
    }
}
