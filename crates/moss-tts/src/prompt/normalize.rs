//! Fixed OpenMOSS robustness normalization; no semantic number expansion.
//! Apache-2.0 upstream source identity is recorded in the crate SOURCE.md.
mod unicode13;
use fancy_regex::{Captures, Regex};
use std::sync::LazyLock;

type Rules = Vec<(Regex, &'static str)>;
const CJK: &str = r"[\u{3400}-\u{4dbf}\u{4e00}-\u{9fff}\u{3040}-\u{30ff}]";
const LATIN: &str =
    r"(?:___PROT\d+___|(?=[A-Za-z0-9._/+:-]*[A-Za-z])[A-Za-z0-9][A-Za-z0-9._/+:-]*)";

fn regex(pattern: &str) -> Regex {
    Regex::new(pattern).expect("fixed OpenMOSS regex")
}

fn substitute(
    input: &str,
    pattern: &Regex,
    mut replacement: impl FnMut(&Captures<'_>) -> String,
) -> anyhow::Result<String> {
    let mut output = String::with_capacity(input.len());
    let mut cursor = 0;
    for captures in pattern.captures_iter(input) {
        let captures = captures?;
        let matched = captures.get(0).expect("whole regex match");
        output.push_str(&input[cursor..matched.start()]);
        output.push_str(&replacement(&captures));
        cursor = matched.end();
    }
    output.push_str(&input[cursor..]);
    Ok(output)
}

fn apply(mut input: String, rules: &Rules) -> anyhow::Result<String> {
    for (pattern, replacement) in rules {
        input = substitute(&input, pattern, |captures| {
            let mut output = String::new();
            captures.expand(replacement, &mut output);
            output
        })?;
    }
    Ok(input)
}

fn spaces(input: String) -> anyhow::Result<String> {
    static RULES: LazyLock<Rules> = LazyLock::new(|| {
        vec![
            (regex(r"[ \t\r\f\x0b]+"), " "),
            (regex(&format!(r"({CJK})\s+(?={CJK})")), "$1"),
            (regex(&format!(r"({CJK})\s+(?=\d)")), "$1"),
            (regex(&format!(r"(\d)\s+(?={CJK})")), "$1"),
            (regex(&format!(r"({CJK})(?=({LATIN}))")), "$1 "),
            (regex(&format!(r"(({LATIN}))(?={CJK})")), "$1 "),
            (regex(r" {2,}"), " "),
            (regex(r"\s+([，。！？；：、”’」』】）》])"), "$1"),
            (regex(r"([（【「『《“‘])\s+"), "$1"),
            (regex(r"([，。！？；：、])\s*"), "$1"),
            (regex(r"\s+([,.;!?])"), "$1"),
            (regex(r" {2,}"), " "),
        ]
    });
    Ok(apply(input, &RULES)?.trim().to_owned())
}

pub(super) fn text(input: &str) -> anyhow::Result<String> {
    static MARKDOWN_LINK: LazyLock<Rules> =
        LazyLock::new(|| vec![(regex(r"\[([^\[\]]+?)\]\((https?://[^)\s]+)\)"), "$1 $2")]);
    static LINE_PREFIXES: LazyLock<Rules> = LazyLock::new(|| {
        [r"^#{1,6}\s+", r"^>\s+", r"^[-*+]\s+", r"^\d+[.)]\s+"]
            .into_iter()
            .map(|pattern| (regex(pattern), ""))
            .collect()
    });
    static PROTECTED: LazyLock<Vec<Regex>> = LazyLock::new(|| {
        [
            r"https?://[^\s\u{3000}，。！？；、）】》〉」』]+",
            // Python's Unicode \w is letters/numbers/underscore, excluding marks.
            r"(?<![\p{L}\p{N}_.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?![\p{L}\p{N}_.-])",
            r"(?<![A-Za-z0-9_])@[A-Za-z0-9_]{1,32}",
            r"(?<![A-Za-z0-9_])(?:u|r)/[A-Za-z0-9_]+",
            r"(?<![A-Za-z0-9_])#(?!\s)[^\s#]+",
            r"(?<![A-Za-z0-9_])\.(?=[A-Za-z0-9._-]*[A-Za-z0-9])[A-Za-z0-9._-]+",
            r"(?<![A-Za-z0-9_])(?=[A-Za-z0-9._/+:-]*[A-Za-z])(?=[A-Za-z0-9._/+:-]*[._/+:-])[A-Za-z0-9](?:[A-Za-z0-9._/+:-]*[A-Za-z0-9])?(?![A-Za-z0-9_])",
        ].into_iter().map(regex).collect()
    });
    static STRUCTURAL: LazyLock<Rules> = LazyLock::new(|| {
        let block = r"(^|[。！？!?；;]\s*)[【〖『「]([^】〗』」]+)[】〗』」]\s*";
        vec![
            (regex(block), "$1$2。"),
            (regex(block), "$1$2。"),
            (
                regex(
                    r"(^|[。！？!?；;]\s*)《([^》]+)》(?=\s*(?:___PROT\d+___|[—–―-]{2,}|$|[。！？!?；;，,]))",
                ),
                "$1$2",
            ),
            (
                regex(r"\s*(?:<[-=]+>|[-=]+>|<[-=]+|[→←↔⇒⇐⇔⟶⟵⟷⟹⟸⟺↦↤↪↩])\s*"),
                "，",
            ),
            (regex(r"\s*(?:—|–|―|-){2,}\s*"), "。"),
        ]
    });
    static REPEATED: LazyLock<Rules> = LazyLock::new(|| {
        [
            (r"(?:\.{3,}|…{2,}|……+)", "。"),
            (r"[。．]{2,}", "。"),
            (r"[，,]{2,}", "，"),
            (r"[!！]{2,}", "！"),
            (r"[?？]{2,}", "？"),
        ]
        .into_iter()
        .map(|(pattern, value)| (regex(pattern), value))
        .collect()
    });
    static MIXED_QE: LazyLock<Regex> = LazyLock::new(|| regex(r"[!?！？]{2,}"));

    let input = input
        .replace("\r\n", "\n")
        .replace('\r', "\n")
        .replace('\u{3000}', " ");
    // Match the pinned reference environment's Unicode categories, including
    // characters that were unassigned there but added to newer Unicode releases.
    let input: String = input
        .chars()
        .filter(|&character| {
            if matches!(character, '\n' | '\t' | ' ') {
                return true;
            }
            let code = character as u32;
            let index = unicode13::OTHER_RANGES.partition_point(|&(start, _)| start <= code);
            index == 0 || code > unicode13::OTHER_RANGES[index - 1].1
        })
        .collect();
    let input = apply(input, &MARKDOWN_LINK)?;
    let lines = input
        .split(['\n', '\u{2028}', '\u{2029}'])
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(|line| apply(line.to_owned(), &LINE_PREFIXES))
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut input = lines.join("。");
    let mut protected = Vec::new();
    for pattern in PROTECTED.iter() {
        input = substitute(&input, pattern, |captures| {
            let placeholder = format!("___PROT{}___", protected.len());
            protected.push(captures.get(0).expect("whole match").as_str().to_owned());
            placeholder
        })?;
    }
    input = spaces(input)?;
    input = apply(input, &STRUCTURAL)?;
    input = apply(input, &REPEATED)?;
    input = substitute(&input, &MIXED_QE, |captures| {
        let value = captures.get(0).expect("whole match").as_str();
        match (value.contains(['?', '？']), value.contains(['!', '！'])) {
            (true, true) => "？！",
            (true, false) => "？",
            _ => "！",
        }
        .into()
    })?;
    input = spaces(input)?;
    for (index, original) in protected.iter().enumerate() {
        input = input.replace(&format!("___PROT{index}___"), original);
    }
    Ok(input.trim().to_owned())
}

#[cfg(test)]
mod tests {
    #[test]
    fn fixed_official_normalization_cases_match() -> anyhow::Result<()> {
        let fixture: serde_json::Value = serde_json::from_str(include_str!(
            "../../tests/fixtures/voicedesign-normalization.json"
        ))?;
        for case in fixture["cases"].as_array().unwrap() {
            assert_eq!(
                super::text(case["text"].as_str().unwrap())?,
                case["normalized"].as_str().unwrap(),
                "{}",
                case["id"]
            );
        }
        Ok(())
    }
}
