//! Fixed IndexTTS text protections, annotations and resource-backed normalization.
//! Japanese segmentation lives in `japanese`; CN/EN normalization uses the pinned
//! WeText FSTs. Missing resources fail explicitly; no approximate number fallback.

use std::collections::HashMap;
use std::sync::OnceLock;

use candle_core::Result;
use regex::Regex;
use std::{
    cell::RefCell,
    path::{Path, PathBuf},
};
mod nemo;
mod normalization;

pub(crate) mod japanese;

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

/// Fixed text-normalization resources resolved individually by the host.
pub struct TextResources {
    pub normalizer: [PathBuf; 4],
    pub spanish: [PathBuf; 2],
    pub linux: bool,
}

enum NormalizationResources {
    Directory(PathBuf),
    Files(TextResources),
}

pub struct TextFrontend {
    glossary: Vec<(Regex, String)>,
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
    normalization_resources: Option<NormalizationResources>,
    normalization: RefCell<Option<normalization::Normalization>>,
    spanish: RefCell<Option<normalization::Spanish>>,
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
            normalization_resources: None,
            normalization: RefCell::new(None),
            spanish: RefCell::new(None),
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
        entries.sort_by(|(a, _), (b, _)| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));
        self.glossary = entries
            .into_iter()
            .map(|(term, reading)| {
                let re =
                    Regex::new(&format!("(?i){}", regex::escape(&term))).expect("escaped glossary");
                (re, reading)
            })
            .collect();
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
        self.pinyin_re
            .find_iter(s)
            .any(|m| !self.prev_is_ascii_letter(s, m.start()))
    }

    fn prev_is_ascii_letter(&self, s: &str, byte_start: usize) -> bool {
        s[..byte_start]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_ascii_alphabetic())
    }

    /// The fixed inference frontend order, before token-budget segmentation.
    #[cfg(test)]
    pub(crate) fn prepare(
        &self,
        text: &str,
        language: &str,
        normalize: bool,
        japanese: Option<&japanese::JapaneseFrontend>,
    ) -> Result<String> {
        self.prepare_with_cancel(text, language, normalize, japanese, &|| false)
    }

    pub(crate) fn prepare_with_cancel(
        &self,
        text: &str,
        language: &str,
        normalize: bool,
        japanese: Option<&japanese::JapaneseFrontend>,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<String> {
        if cancelled() {
            candle_core::bail!("IndexTTS text processing cancelled");
        }
        let mut text = self.clean_pattern(text);
        if normalize && matches!(language, "zh" | "zhen" | "en") {
            text = self.normalize_with_cancel(&text, cancelled)?;
        }
        if normalize && language == "es" {
            let mut spanish = self.spanish.borrow_mut();
            if spanish.is_none() {
                let resources = self.normalization_resources.as_ref().ok_or_else(|| {
                    candle_core::Error::Msg("IndexTTS TN resources were not configured".into())
                })?;
                *spanish = Some(match resources {
                    NormalizationResources::Directory(root) => {
                        normalization::Spanish::load(&root.with_file_name("es"))?
                    }
                    NormalizationResources::Files(files) => {
                        normalization::Spanish::load_files(&files.spanish[0], &files.spanish[1])?
                    }
                });
            }
            text = spanish
                .as_ref()
                .ok_or_else(|| candle_core::Error::Msg("Spanish TN was not loaded".into()))?
                .normalize_with_cancel(&text, cancelled)?;
        }
        if matches!(language, "ja" | "zh" | "zhen" | "en") {
            text = text.to_lowercase();
        }
        if language == "es" {
            text = text.to_uppercase();
        }
        text = apply_pronunciation_annotations(&text);
        if language == "ja" {
            let japanese = japanese.ok_or_else(|| {
                candle_core::Error::Msg("IndexTTS Japanese dictionary is not prepared".into())
            })?;
            text = japanese.process_with_cancel(&text, cancelled)?;
        }
        Ok(restore_special_token_case(&text))
    }

    /// Configure the fixed FST directory. Resources load lazily when CN/EN TN
    /// is requested; Japanese does not run these normalizers.
    pub fn with_resources(directory: &Path) -> Self {
        Self {
            normalization_resources: Some(NormalizationResources::Directory(directory.to_owned())),
            ..Self::new()
        }
    }

    /// Configure the fixed normalization graphs as independently resolved files.
    pub fn with_files(files: TextResources) -> Self {
        Self {
            normalization_resources: Some(NormalizationResources::Files(files)),
            ..Self::new()
        }
    }

    /// Normalize using the official automatic CN/EN text route, independently
    /// of the acoustic language embedding. Resource failures are explicit.
    pub fn normalize(&self, text: &str) -> Result<String> {
        self.normalize_with_cancel(text, &|| false)
    }

    /// Normalize with the owning request's cancellation signal.
    pub fn normalize_with_cancel(
        &self,
        text: &str,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<String> {
        if cancelled() {
            candle_core::bail!("IndexTTS text processing cancelled");
        }
        let mut normalization = self.normalization.borrow_mut();
        if normalization.is_none() {
            let resources = self.normalization_resources.as_ref().ok_or_else(|| {
                candle_core::Error::Msg("IndexTTS TN resources were not configured".into())
            })?;
            *normalization = Some(match resources {
                NormalizationResources::Directory(root) => {
                    normalization::Normalization::load(root)?
                }
                NormalizationResources::Files(files) => normalization::Normalization::load_files(
                    [
                        &files.normalizer[0],
                        &files.normalizer[1],
                        &files.normalizer[2],
                        &files.normalizer[3],
                    ],
                    files.linux,
                )?,
            });
        }
        let normalization = normalization.as_mut().ok_or_else(|| {
            candle_core::Error::Msg("IndexTTS TN resources were not loaded".into())
        })?;
        self.normalize_with(text, |text, chinese| {
            normalization.normalize_with_cancel(text, chinese, cancelled)
        })
    }

    fn normalize_with(
        &self,
        text: &str,
        mut normalize: impl FnMut(&str, bool) -> Result<String>,
    ) -> Result<String> {
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

        let zh = self.use_chinese(&protected);
        protected = self
            .contraction_re
            .replace_all(&protected, "${1} is")
            .to_string();

        for (re, reading) in &self.glossary {
            protected = re.replace_all(&protected, reading.as_str()).to_string();
        }

        let result = if zh {
            let (step1, tech) = self.save_tech_terms(protected.trim_end());
            let (step2, pinyin) = self.save_pinyin_tones(&step1);
            let (step3, names) = self.save_names(&step2);
            let mut normalized = normalize(&step3, true)?;
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
            let normalized = normalize(&step1, false)?;
            let normalized = self.restore_tech_terms(normalized, &tech);
            self.fold(&normalized, &self.base_map)
        };

        let mut out = result;
        for (key, original) in placeholders {
            out = out.replace(&key, &original);
        }
        Ok(out)
    }

    fn save_tech_terms(&self, text: &str) -> (String, Vec<String>) {
        let mut terms: Vec<String> = self
            .tech_re
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .collect();
        dedup_longest_first(&mut terms);
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
        let mut found: Vec<String> = self
            .pinyin_re
            .find_iter(text)
            .filter(|m| !self.prev_is_ascii_letter(text, m.start()))
            .map(|m| m.as_str().to_string())
            .collect();
        if found.is_empty() {
            return (text.to_string(), None);
        }
        dedup_longest_first(&mut found);
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
        dedup_longest_first(&mut names);
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

/// Deduplicate and order by descending byte length so substring replacements
/// never corrupt longer matches containing shorter ones.
fn dedup_longest_first(items: &mut Vec<String>) {
    items.sort();
    items.dedup();
    items.sort_by_key(|s| std::cmp::Reverse(s.len()));
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
    static JQX_RE: OnceLock<Regex> = OnceLock::new();
    let first = pinyin.chars().next().unwrap_or(' ');
    if !"jqxJQX".contains(first) {
        return pinyin.to_string();
    }
    let re = JQX_RE.get_or_init(|| Regex::new(r"(?i)([jqx])[uü](n|e|an)*(\d)").expect("static"));
    re.replace(pinyin, "${1}v${2}${3}").to_uppercase()
}

/// `<文字|发音>` → `<|SPECIAL_TOKEN_1|>/<|SPECIAL_TOKEN_2|>` wrapping with
/// kana passthrough (`apply_pronunciation_annotations` in `infer_v2_5.py`).
pub fn apply_pronunciation_annotations(text: &str) -> String {
    static PRON_RE: OnceLock<Regex> = OnceLock::new();
    static HIRA_RE: OnceLock<Regex> = OnceLock::new();
    static KATA_RE: OnceLock<Regex> = OnceLock::new();
    static ZH_RE: OnceLock<Regex> = OnceLock::new();
    let pron_re = PRON_RE.get_or_init(|| Regex::new(r"<([^|>\n]+)\|([^>\n]+)>").expect("static"));
    let hira = HIRA_RE.get_or_init(|| Regex::new(r"^[\x{3040}-\x{309F}]+$").expect("static"));
    let kata = KATA_RE.get_or_init(|| Regex::new(r"^[\x{30A0}-\x{30FF}]+$").expect("static"));
    let chinese = ZH_RE.get_or_init(|| Regex::new(r"[\x{4e00}-\x{9fff}]").expect("static"));
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

/// Restore special-token case after language-specific preprocessing, matching
/// the final substitution in the fixed `infer_v2_5.py` text path.
pub(crate) fn restore_special_token_case(text: &str) -> String {
    static SPECIAL_RE: OnceLock<Regex> = OnceLock::new();
    let re = SPECIAL_RE.get_or_init(|| Regex::new(r"<\|([^|]+)\|>").expect("static regex"));
    re.replace_all(text, |caps: &regex::Captures| {
        format!("<|{}|>", caps[1].to_uppercase())
    })
    .into_owned()
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
    fn normalization_requires_explicit_resources() {
        assert!(TextFrontend::new().normalize("2026年").is_err());
    }

    #[test]
    #[cfg(not(target_os = "linux"))]
    #[ignore = "requires pinned Darwin WeText/Spanish FSTs and UniDic dictionary"]
    fn complete_public_matrix_matches_official_frontend() {
        let root = std::env::var_os("TALECHIME_INDEXTTS_WETEXT").expect("fixed TN directory");
        let dictionary = std::env::var_os("TALECHIME_INDEXTTS_JAPANESE_DICTIONARY")
            .expect("fixed UniDic dictionary");
        let frontend = TextFrontend::with_resources(Path::new(&root));
        let japanese = japanese::JapaneseFrontend::load(Path::new(&dictionary)).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/public-matrix-oracle.json"))
                .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            if case.get("expected_rejection").is_some() {
                continue; // The adapter's five-language API rejects this before the frontend.
            }
            assert_eq!(
                frontend
                    .prepare(
                        case["text"].as_str().unwrap(),
                        case["language"].as_str().unwrap(),
                        true,
                        Some(&japanese)
                    )
                    .unwrap(),
                case["output"].as_str().unwrap(),
                "{}",
                case["id"]
            );
        }
    }

    #[test]
    #[ignore = "requires explicit pinned WeText FST directory"]
    fn official_tn_oracle() {
        let root = std::env::var_os("TALECHIME_INDEXTTS_WETEXT").expect("fixed FST directory");
        let frontend = TextFrontend::with_resources(Path::new(&root));
        let fixture: serde_json::Value = serde_json::from_str(if cfg!(target_os = "linux") {
            include_str!("../tests/fixtures/tn-linux-oracle.json")
        } else {
            include_str!("../tests/fixtures/tn-macos-oracle.json")
        })
        .unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            let cleaned = frontend.clean_pattern(text);
            assert_eq!(
                frontend.normalize(&cleaned).unwrap(),
                case["output"].as_str().unwrap(),
                "{text:?}"
            );
        }
    }

    #[test]
    #[ignore = "requires explicit fixed CN/EN and Spanish FST directories"]
    fn both_platforms_and_spanish_match_independent_official_oracles() {
        let root =
            std::env::var_os("TALECHIME_INDEXTTS_WETEXT").expect("fixed tn/wetext directory");
        let root = Path::new(&root);
        for (linux, fixture) in [
            (
                false,
                include_str!("../tests/fixtures/tn-macos-oracle.json"),
            ),
            (true, include_str!("../tests/fixtures/tn-linux-oracle.json")),
        ] {
            let frontend = TextFrontend::new();
            let mut tn = normalization::Normalization::load_platform(root, linux).unwrap();
            let fixture: serde_json::Value = serde_json::from_str(fixture).unwrap();
            for case in fixture["cases"].as_array().unwrap() {
                let text = case["text"].as_str().unwrap();
                let cleaned = frontend.clean_pattern(text);
                let actual = frontend
                    .normalize_with(&cleaned, |text, chinese| tn.normalize(text, chinese))
                    .unwrap();
                assert_eq!(
                    actual,
                    case["output"].as_str().unwrap(),
                    "linux={linux} {text:?}"
                );
            }
        }
        compare_spanish_oracle(root);
    }

    #[test]
    #[ignore = "requires separately staged fixed WeText and IndexTTS frontend files"]
    fn independent_graph_paths_match_official_oracles() {
        let wetext =
            PathBuf::from(std::env::var_os("TALECHIME_WETEXT_FILES").expect("fixed WeText files"));
        let index = PathBuf::from(
            std::env::var_os("TALECHIME_INDEX_FRONTEND_FILES").expect("fixed IndexTTS files"),
        );
        for (linux, fixture) in [
            (
                false,
                include_str!("../tests/fixtures/tn-macos-oracle.json"),
            ),
            (true, include_str!("../tests/fixtures/tn-linux-oracle.json")),
        ] {
            let normalizer = if linux {
                [
                    "zh/tagger.fst",
                    "zh/verbalizer.fst",
                    "en/tagger.fst",
                    "en/verbalizer.fst",
                ]
                .map(|name| index.join("tn/linux").join(name))
            } else {
                [
                    "zh/tn/tagger.fst",
                    "zh/tn/verbalizer.fst",
                    "en/tn/tagger.fst",
                    "en/tn/verbalizer.fst",
                ]
                .map(|name| wetext.join(name))
            };
            let frontend = TextFrontend::with_files(TextResources {
                normalizer,
                spanish: [
                    index.join("tn/es/tagger.fst"),
                    index.join("tn/es/verbalizer.fst"),
                ],
                linux,
            });
            let fixture: serde_json::Value = serde_json::from_str(fixture).unwrap();
            for case in fixture["cases"].as_array().unwrap() {
                let input = case["text"].as_str().unwrap();
                assert_eq!(
                    frontend.normalize(&frontend.clean_pattern(input)).unwrap(),
                    case["output"].as_str().unwrap(),
                    "linux={linux} {input:?}"
                );
            }
            let spanish = normalization::Spanish::load_files(
                &index.join("tn/es/tagger.fst"),
                &index.join("tn/es/verbalizer.fst"),
            )
            .unwrap();
            let cases: serde_json::Value =
                serde_json::from_str(include_str!("../tests/fixtures/spanish-oracle.json"))
                    .unwrap();
            for case in cases["cases"].as_array().unwrap() {
                let input = case["text"].as_str().unwrap();
                assert_eq!(
                    spanish.normalize(input).unwrap(),
                    case["output"].as_str().unwrap(),
                    "{input:?}"
                );
            }
        }
    }

    fn compare_spanish_oracle(root: &Path) {
        let spanish = normalization::Spanish::load(&root.with_file_name("es")).unwrap();
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/spanish-oracle.json")).unwrap();
        for case in fixture["cases"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            assert_eq!(
                spanish.normalize(text).unwrap(),
                case["output"].as_str().unwrap(),
                "{text:?}"
            );
        }
    }

    #[test]
    #[ignore = "requires explicit fixed Spanish NeMo FST directory"]
    fn spanish_matches_independent_official_oracle() {
        let root =
            std::env::var_os("TALECHIME_INDEXTTS_WETEXT").expect("fixed tn/wetext directory");
        compare_spanish_oracle(Path::new(&root));
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
        assert_eq!(
            f.normalize_with("GPT-5 发布", |text, _| Ok(text.replace("5", "五")))
                .unwrap(),
            "GPT-五 发布"
        );
    }

    #[test]
    fn overlapping_pinyin_and_names_preserve_longest_matches() {
        let f = TextFrontend::new();
        // Shorter "an1" must not corrupt "gan1" or "juan1" during placeholder substitution.
        assert_eq!(
            f.normalize_with("读作gan1和an1以及juan1", |text, _| Ok(text.to_owned()))
                .unwrap(),
            "读作gan1和an1以及JVAN1"
        );
        // Shorter "约瑟夫·高登" must not split "约瑟夫·高登-莱维特" before normalizer runs.
        assert_eq!(
            f.normalize_with(
                "约瑟夫·高登-莱维特与约瑟夫·高登2人",
                |text, _| Ok(text.replace("2", "二"))
            )
            .unwrap(),
            "约瑟夫-高登-莱维特与约瑟夫-高登二人"
        );
    }
}
