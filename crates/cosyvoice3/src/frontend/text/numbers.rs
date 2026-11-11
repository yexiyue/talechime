//! Default inflect 7.5.0 cardinal semantics used by the fixed CosyVoice frontend.
//! Unicode digit classification is fixed to the independent Python oracle.
use candle_core::Result;
const DIGITS: &[(u32, u32)] = &[
    (0x30, 0x39),
    (0xb2, 0xb3),
    (0xb9, 0xb9),
    (0x660, 0x669),
    (0x6f0, 0x6f9),
    (0x7c0, 0x7c9),
    (0x966, 0x96f),
    (0x9e6, 0x9ef),
    (0xa66, 0xa6f),
    (0xae6, 0xaef),
    (0xb66, 0xb6f),
    (0xbe6, 0xbef),
    (0xc66, 0xc6f),
    (0xce6, 0xcef),
    (0xd66, 0xd6f),
    (0xde6, 0xdef),
    (0xe50, 0xe59),
    (0xed0, 0xed9),
    (0xf20, 0xf29),
    (0x1040, 0x1049),
    (0x1090, 0x1099),
    (0x1369, 0x1371),
    (0x17e0, 0x17e9),
    (0x1810, 0x1819),
    (0x1946, 0x194f),
    (0x19d0, 0x19da),
    (0x1a80, 0x1a89),
    (0x1a90, 0x1a99),
    (0x1b50, 0x1b59),
    (0x1bb0, 0x1bb9),
    (0x1c40, 0x1c49),
    (0x1c50, 0x1c59),
    (0x2070, 0x2070),
    (0x2074, 0x2079),
    (0x2080, 0x2089),
    (0x2460, 0x2468),
    (0x2474, 0x247c),
    (0x2488, 0x2490),
    (0x24ea, 0x24ea),
    (0x24f5, 0x24fd),
    (0x24ff, 0x24ff),
    (0x2776, 0x277e),
    (0x2780, 0x2788),
    (0x278a, 0x2792),
    (0xa620, 0xa629),
    (0xa8d0, 0xa8d9),
    (0xa900, 0xa909),
    (0xa9d0, 0xa9d9),
    (0xa9f0, 0xa9f9),
    (0xaa50, 0xaa59),
    (0xabf0, 0xabf9),
    (0xff10, 0xff19),
    (0x104a0, 0x104a9),
    (0x10a40, 0x10a43),
    (0x10d30, 0x10d39),
    (0x10e60, 0x10e68),
    (0x11052, 0x1105a),
    (0x11066, 0x1106f),
    (0x110f0, 0x110f9),
    (0x11136, 0x1113f),
    (0x111d0, 0x111d9),
    (0x112f0, 0x112f9),
    (0x11450, 0x11459),
    (0x114d0, 0x114d9),
    (0x11650, 0x11659),
    (0x116c0, 0x116c9),
    (0x11730, 0x11739),
    (0x118e0, 0x118e9),
    (0x11950, 0x11959),
    (0x11c50, 0x11c59),
    (0x11d50, 0x11d59),
    (0x11da0, 0x11da9),
    (0x16a60, 0x16a69),
    (0x16b50, 0x16b59),
    (0x1d7ce, 0x1d7ff),
    (0x1e140, 0x1e149),
    (0x1e2f0, 0x1e2f9),
    (0x1e950, 0x1e959),
    (0x1f100, 0x1f10a),
    (0x1fbf0, 0x1fbf9),
];
const ZEROES: &[u32] = &[
    0x30, 0x660, 0x6f0, 0x7c0, 0x966, 0x9e6, 0xa66, 0xae6, 0xb66, 0xbe6, 0xc66, 0xce6, 0xd66,
    0xde6, 0xe50, 0xed0, 0xf20, 0x1040, 0x1090, 0x17e0, 0x1810, 0x1946, 0x19d0, 0x1a80, 0x1a90,
    0x1b50, 0x1bb0, 0x1c40, 0x1c50, 0xa620, 0xa8d0, 0xa900, 0xa9d0, 0xa9f0, 0xaa50, 0xabf0, 0xff10,
    0x104a0, 0x10d30, 0x11066, 0x110f0, 0x11136, 0x111d0, 0x112f0, 0x11450, 0x114d0, 0x11650,
    0x116c0, 0x11730, 0x118e0, 0x11950, 0x11c50, 0x11d50, 0x11da0, 0x16a60, 0x16b50, 0x1d7ce,
    0x1d7d8, 0x1d7e2, 0x1d7ec, 0x1d7f6, 0x1e140, 0x1e2f0, 0x1e950, 0x1fbf0,
];
const SMALL: [&str; 20] = [
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
const TENS: [&str; 10] = [
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
];
const SCALES: [&str; 12] = [
    "",
    "thousand",
    "million",
    "billion",
    "trillion",
    "quadrillion",
    "quintillion",
    "sextillion",
    "septillion",
    "octillion",
    "nonillion",
    "decillion",
];
fn under_thousand(number: usize) -> String {
    if number >= 100 {
        let head = format!("{} hundred", SMALL[number / 100]);
        return if number.is_multiple_of(100) {
            head
        } else {
            format!("{head} and {}", under_thousand(number % 100))
        };
    }
    if number < 20 {
        return SMALL[number].into();
    }
    if number.is_multiple_of(10) {
        return TENS[number / 10].into();
    }
    format!("{}-{}", TENS[number / 10], SMALL[number % 10])
}
fn cardinal(text: &str) -> Result<String> {
    let digits: String = text
        .chars()
        .filter_map(|ch| {
            ZEROES.iter().find_map(|zero| {
                let code = ch as u32;
                (code >= *zero && code < *zero + 10)
                    .then(|| char::from(b'0' + (code - *zero) as u8))
            })
        })
        .collect();
    let digits = digits.trim_start_matches('0');
    if digits.is_empty() {
        return Ok("zero".into());
    }
    if digits.len() > 36 {
        candle_core::bail!("English cardinal exceeds inflect's decillion range");
    }
    let mut groups = Vec::new();
    for (scale, part) in digits.as_bytes().rchunks(3).enumerate() {
        let number = part.iter().fold(0usize, |value, digit| {
            value * 10 + usize::from(digit - b'0')
        });
        if number > 0 {
            let mut words = under_thousand(number);
            if scale > 0 {
                words.push(' ');
                words.push_str(SCALES[scale]);
            }
            groups.push((scale, number, words));
        }
    }
    groups.reverse();
    let mut output = String::new();
    for (index, (scale, number, words)) in groups.iter().enumerate() {
        if index > 0 {
            output.push_str(if *scale == 0 && *number < 100 {
                " and "
            } else {
                ", "
            });
        }
        output.push_str(words);
    }
    Ok(output)
}
/// Replace contiguous Python isdigit runs, preserving other text verbatim.
pub(super) fn spell_out(text: &str) -> Result<String> {
    let mut output = String::new();
    let mut digits = String::new();
    for ch in text.chars() {
        if DIGITS
            .iter()
            .any(|(start, end)| (*start..=*end).contains(&(ch as u32)))
        {
            digits.push(ch);
        } else {
            if !digits.is_empty() {
                output.push_str(&cardinal(&digits)?);
                digits.clear();
            }
            output.push(ch);
        }
    }
    if !digits.is_empty() {
        output.push_str(&cardinal(&digits)?);
    }
    Ok(output)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cardinal_defaults_match_inflect() {
        for (text, expected) in [
            ("007", "seven"),
            ("2026", "two thousand and twenty-six"),
            ("1001001", "one million, one thousand and one"),
            ("２０２６", "two thousand and twenty-six"),
            ("²", "zero"),
        ] {
            assert_eq!(spell_out(text).unwrap(), expected);
        }
    }
}
