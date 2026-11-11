//! English number words, as misaki gets them from Python's `num2words`.
//!
//! misaki passes digits through `num2words` and then phonemizes the words,
//! so the exact wording ("nineteen oh-five", "one thousand and five")
//! decides what Kokoro says. This reproduces num2words 0.5.14's English
//! output string for string. num2words itself is LGPL and is not used: the
//! test suite checks this against 7,026 of its outputs
//! (`tests/data/num2words-en.tsv`).

const ONES: [&str; 20] = [
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
const TENS: [&str; 10] = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
const SCALES: [&str; 7] = ["", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion"];

/// Cardinal words: 1234 is "one thousand, two hundred and thirty-four".
pub fn cardinal(n: u64) -> String {
    if n == 0 {
        return ONES[0].to_owned();
    }
    // Thousands groups, most significant first, with their scale index.
    let mut groups = Vec::new();
    let mut rest = n;
    let mut scale = 0;
    while rest > 0 {
        let group = rest % 1000;
        if group != 0 {
            groups.push((group, scale));
        }
        rest /= 1000;
        scale += 1;
    }
    groups.reverse();

    let mut out = String::new();
    let last = groups.len() - 1;
    for (i, &(group, scale)) in groups.iter().enumerate() {
        if i > 0 {
            // A trailing group under 100 joins with "and"; the rest with commas.
            out.push_str(if i == last && scale == 0 && group < 100 { " and " } else { ", " });
        }
        out.push_str(&below_thousand(group));
        if scale > 0 {
            out.push(' ');
            out.push_str(SCALES[scale]);
        }
    }
    out
}

fn below_thousand(n: u64) -> String {
    let (hundreds, rest) = (n / 100, n % 100);
    match (hundreds, rest) {
        (0, r) => below_hundred(r),
        (h, 0) => format!("{} hundred", ONES[h as usize]),
        (h, r) => format!("{} hundred and {}", ONES[h as usize], below_hundred(r)),
    }
}

fn below_hundred(n: u64) -> String {
    match n {
        0..=19 => ONES[n as usize].to_owned(),
        _ if n.is_multiple_of(10) => TENS[(n / 10) as usize].to_owned(),
        _ => format!("{}-{}", TENS[(n / 10) as usize], ONES[(n % 10) as usize]),
    }
}

/// Ordinal words: 21 is "twenty-first", 100 is "one hundredth".
pub fn ordinal(n: u64) -> String {
    let words = cardinal(n);
    let cut = words.rfind([' ', '-']).map_or(0, |i| i + 1);
    let (head, last) = words.split_at(cut);
    let last = match last {
        "one" => "first".to_owned(),
        "two" => "second".to_owned(),
        "three" => "third".to_owned(),
        "five" => "fifth".to_owned(),
        "eight" => "eighth".to_owned(),
        "nine" => "ninth".to_owned(),
        "twelve" => "twelfth".to_owned(),
        w if w.ends_with('y') => format!("{}ieth", &w[..w.len() - 1]),
        w => format!("{w}th"),
    };
    format!("{head}{last}")
}

/// Year words: 1905 is "nineteen oh-five", 2010 "twenty ten", but 2005
/// stays "two thousand and five".
pub fn year(n: u64) -> String {
    let (high, low) = (n / 100, n % 100);
    if high == 0 || (high % 10 == 0 && low < 10) || high >= 100 {
        return cardinal(n);
    }
    let low = match low {
        0 => "hundred".to_owned(),
        1..=9 => format!("oh-{}", cardinal(low)),
        _ => cardinal(low),
    };
    format!("{} {low}", cardinal(high))
}

/// Decimal words for a non-negative decimal literal such as `"4.70"`:
/// "four point seven". The fraction is read digit by digit, trailing zeros
/// dropped as Python's float repr drops them; a whole value ("5.0") reads
/// as the integer. Returns `None` for anything that is not digits, an
/// optional point and digits.
pub fn decimal(literal: &str) -> Option<String> {
    let (whole, fraction) = literal.split_once('.').unwrap_or((literal, ""));
    let whole = if whole.is_empty() { "0" } else { whole };
    if !whole.bytes().all(|b| b.is_ascii_digit()) || !fraction.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let mut out = cardinal(whole.parse().ok()?);
    let fraction = fraction.trim_end_matches('0');
    if !fraction.is_empty() {
        out.push_str(" point");
        for digit in fraction.bytes() {
            out.push(' ');
            out.push_str(ONES[(digit - b'0') as usize]);
        }
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_every_num2words_oracle_vector() {
        let oracle = include_str!("../tests/data/num2words-en.tsv");
        let mut checked = 0;
        for line in oracle.lines().filter(|l| !l.starts_with('#')) {
            let mut cols = line.split('\t');
            let (kind, input, expected) = (cols.next().unwrap(), cols.next().unwrap(), cols.next().unwrap());
            let got = match kind {
                "cardinal" => cardinal(input.parse().unwrap()),
                "ordinal" => ordinal(input.parse().unwrap()),
                "year" => year(input.parse().unwrap()),
                "float" => decimal(input).unwrap(),
                other => panic!("unknown kind {other}"),
            };
            assert_eq!(got, expected, "{kind} {input}");
            checked += 1;
        }
        assert_eq!(checked, 7026);
    }

    #[test]
    fn rejects_non_numeric_decimals() {
        assert_eq!(decimal("1.2.3"), None);
        assert_eq!(decimal("12a"), None);
        assert_eq!(decimal(".5").as_deref(), Some("zero point five"));
    }
}
