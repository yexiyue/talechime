use super::VerificationError;
use chinese_number::{ChineseCountMethod, ChineseToNumber};
use unicode_normalization::UnicodeNormalization;
pub(super) struct Normalizer(ferrous_opencc::OpenCC);
impl Normalizer {
    pub(super) fn new() -> Result<Self, VerificationError> {
        ferrous_opencc::OpenCC::from_config(ferrous_opencc::config::BuiltinConfig::T2s)
            .map(Self)
            .map_err(|error| VerificationError::Invalid(format!("readback dictionary: {error}")))
    }
    pub(super) fn normalize(&self, text: &str) -> String {
        let text = self
            .0
            .convert(&text.nfkc().flat_map(char::to_lowercase).collect::<String>());
        let mut output = String::new();
        let mut run = String::new();
        fn flush(run: &mut String, output: &mut String) {
            if run.is_empty() {
                return;
            }
            // Only explicit digit strings or complete positional forms are collapsed.
            // Colloquial shorthand such as 一百二 remains ambiguous.
            let units = run.contains(['十', '百', '千', '万', '亿']);
            if !units {
                // A spoken digit sequence carries its width, including fractional zeros.
                for digit in run.chars() {
                    output.push(match digit {
                        '零' | '〇' => '0',
                        '一' => '1',
                        '二' | '两' => '2',
                        '三' => '3',
                        '四' => '4',
                        '五' => '5',
                        '六' => '6',
                        '七' => '7',
                        '八' => '8',
                        '九' => '9',
                        _ => unreachable!("digit run contains only Chinese numerals"),
                    });
                }
                run.clear();
                return;
            }
            let complete = run.ends_with(['十', '百', '千', '万', '亿'])
                || run
                    .chars()
                    .rev()
                    .nth(1)
                    .is_some_and(|c| c == '十' || c == '零');
            let value: Result<u64, _> = run.to_number(ChineseCountMethod::TenThousand);
            if complete && let Ok(value) = value {
                output.push_str(&value.to_string());
            } else {
                output.push_str(run);
            }
            run.clear();
        }
        for c in text.chars() {
            if "零〇一二三四五六七八九十百千万亿两".contains(c) {
                run.push(c);
            } else {
                flush(&mut run, &mut output);
                output.push(c);
            }
        }
        flush(&mut run, &mut output);
        let chars: Vec<char> = output.chars().collect();
        chars
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| {
                // Decimal points and signs are content, not ignorable punctuation.
                let next = chars.get(i + 1).is_some_and(|c| c.is_ascii_digit());
                let previous = i > 0 && chars[i - 1].is_ascii_digit();
                if c == '点' && previous && next {
                    return Some('.');
                }
                if c.is_alphanumeric() {
                    return Some(c);
                }
                match c {
                    '.' | '点' if previous && next => Some('.'),
                    '-' | '−' if next => Some('-'),
                    '%' if previous => Some('%'),
                    _ => None,
                }
            })
            .collect()
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn simple_forms_are_equal_but_ambiguity_and_decimal_meaning_are_kept() {
        let n = Normalizer::new().unwrap();
        for (a, b) in [
            ("銀行與歸途", "银行与归途"),
            ("二零二六年一百二十八元", "2026年128元"),
            ("ＡＢＣ，１２３！", "abc123"),
            ("一点五", "1.5"),
        ] {
            assert_eq!(n.normalize(a), n.normalize(b));
        }
        assert_ne!(n.normalize("1.5"), n.normalize("15"));
        assert_ne!(n.normalize("一百二"), n.normalize("120"));
        assert_ne!(n.normalize("-3"), n.normalize("3"));
        assert_ne!(n.normalize("20%"), n.normalize("20"));
    }
    #[test]
    fn spoken_digit_sequences_preserve_leading_and_fractional_zeros() {
        let n = Normalizer::new().unwrap();
        for (spoken, written) in [
            ("一点零五", "1.05"),
            ("零点零零五", "0.005"),
            ("一百二十八点零五", "128.05"),
            ("零零七", "007"),
            ("二零二六年", "2026年"),
        ] {
            assert_eq!(n.normalize(spoken), n.normalize(written));
        }
        assert_ne!(n.normalize("一点零五"), n.normalize("1.5"));
        assert_ne!(n.normalize("零点零零五"), n.normalize("0.05"));
        assert_ne!(n.normalize("零零七"), n.normalize("7"));
    }
}
