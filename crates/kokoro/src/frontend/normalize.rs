//! Mandarin numeric normalization following cn2an 0.5.24's an2cn transform.
use candle_core::Result;
use regex::Regex;

const DIGITS: [char; 10] = ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九'];
const UNITS: [&str; 16] = [
    "", "十", "百", "千", "万", "十", "百", "千", "亿", "十", "百", "千", "万", "十", "百", "千",
];
const MEASURES: &str = "斤|克|千克|公斤|吨|米|厘米|毫米|公里|升|毫升|元|角|分|个|只|条|张|块|瓶|杯|份|本|辆|台|匹|头|位|亩|小时|分钟|秒|天|半";

/// Decimal digits accepted by cn2an: ASCII and full-width, without folding
/// punctuation or unrelated Unicode numeric characters.
pub fn decimal_digit(character: char) -> Option<u8> {
    match character {
        '0'..='9' => Some(character as u8 - b'0'),
        '０'..='９' => Some((character as u32 - '０' as u32) as u8),
        _ => None,
    }
}

fn replace<F>(text: &str, pattern: &str, mut convert: F) -> Result<String>
where
    F: FnMut(&str) -> Result<String>,
{
    let pattern = Regex::new(pattern).map_err(candle_core::Error::wrap)?;
    let mut output = String::new();
    let mut cursor = 0;
    for matched in pattern.find_iter(text) {
        output.push_str(&text[cursor..matched.start()]);
        output.push_str(&convert(matched.as_str())?);
        cursor = matched.end();
    }
    output.push_str(&text[cursor..]);
    Ok(output)
}

fn integer(value: &str) -> Result<String> {
    let value = value.trim_start_matches('0');
    if value.is_empty() {
        return Ok("零".into());
    }
    if value.len() > UNITS.len() {
        candle_core::bail!("Mandarin number exceeds cn2an's 16 digit integer limit");
    }
    let mut output = String::new();
    for (index, byte) in value.bytes().enumerate() {
        let position = value.len() - index - 1;
        let digit = (byte - b'0') as usize;
        if digit != 0 || position.is_multiple_of(4) {
            output.push(DIGITS[digit]);
            output.push_str(UNITS[position]);
        }
        if digit == 0 && index > 0 && !output.ends_with('零') {
            output.push('零');
        }
    }
    output = output
        .replace("零零", "零")
        .replace("零万", "万")
        .replace("零亿", "亿")
        .replace("亿万", "亿")
        .trim_matches('零')
        .to_owned();
    let pattern =
        Regex::new(r"([万亿])零([一二三四五六七八九]千)").map_err(candle_core::Error::wrap)?;
    output = pattern.replace_all(&output, "${1}${2}").into_owned();
    if output.starts_with("一十") {
        output.remove(0);
    }
    if output.is_empty() {
        output.push('零');
    }
    Ok(output)
}

fn number(value: &str) -> Result<String> {
    let (sign, value) = match value.strip_prefix('-') {
        Some(value) => ("负", value),
        None => ("", value),
    };
    let (whole, decimal) = value
        .split_once('.')
        .map_or((value, None), |(a, b)| (a, Some(b)));
    let mut result = format!("{sign}{}", integer(whole)?);
    if let Some(decimal) = decimal {
        if decimal.len() > 16 {
            candle_core::bail!(
                "Mandarin decimal exceeds cn2an precision; refusing silent truncation"
            );
        }
        result.push('点');
        for byte in decimal.bytes() {
            result.push(DIGITS[(byte - b'0') as usize]);
        }
    }
    Ok(result)
}

pub(super) fn normalize(text: &str) -> Result<String> {
    let text: String = text
        .chars()
        .map(|character| {
            decimal_digit(character).map_or(character, |digit| char::from(b'0' + digit))
        })
        .collect();
    let mut text = replace(
        &text,
        &format!(r"[0-9]+(?:\.[0-9]+)?-[0-9]+(?:\.[0-9]+)?(?:{MEASURES})"),
        |value| {
            let end = value
                .find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-')
                .ok_or_else(|| candle_core::Error::Msg("invalid numeric range".into()))?;
            let (a, b) = value[..end]
                .split_once('-')
                .ok_or_else(|| candle_core::Error::Msg("invalid numeric range".into()))?;
            Ok(format!("{}到{}{}", number(a)?, number(b)?, &value[end..]))
        },
    )?;
    text = replace(&text, r"[0-9]{2,4}年", |value| {
        Ok(value[..value.len() - '年'.len_utf8()]
            .bytes()
            .map(|b| DIGITS[(b - b'0') as usize])
            .collect::<String>()
            + "年")
    })?;
    text = replace(&text, r"[0-9]{1,2}[月日]", |value| {
        let boundary = value
            .char_indices()
            .last()
            .map(|(i, _)| i)
            .ok_or_else(|| candle_core::Error::Msg("empty date".into()))?;
        Ok(number(&value[..boundary])? + &value[boundary..])
    })?;
    text = replace(&text, r"[0-9]+/[0-9]+", |value| {
        let (a, b) = value
            .split_once('/')
            .ok_or_else(|| candle_core::Error::Msg("invalid fraction".into()))?;
        Ok(format!("{}分之{}", number(b)?, number(a)?))
    })?;
    text = replace(&text, r"-?(?:[0-9]+\.)?[0-9]+%", |value| {
        Ok(format!("百分之{}", number(&value[..value.len() - 1])?))
    })?;
    text = replace(&text, r"[0-9]+℃", |value| {
        Ok(number(value.trim_end_matches('℃'))? + "摄氏度")
    })?;
    text = replace(&text, r"-?(?:[0-9]+\.)?[0-9]+", number)?;
    for (from, to) in [
        ("、", ", "),
        ("，", ", "),
        ("。", ". "),
        ("．", ". "),
        ("！", "! "),
        ("：", ": "),
        ("；", "; "),
        ("？", "? "),
        ("«", " “"),
        ("»", "” "),
        ("《", " “"),
        ("》", "” "),
        ("「", " “"),
        ("」", "” "),
        ("【", " “"),
        ("】", "” "),
        ("（", " ("),
        ("）", ") "),
    ] {
        text = text.replace(from, to);
    }
    Ok(text.trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dates_units_and_decimals_preserve_official_readings() -> Result<()> {
        assert_eq!(
            normalize("今天是2026年10月27日，价格是123.45元。")?,
            "今天是二零二六年十月二十七日, 价格是一百二十三点四五元."
        );
        assert_eq!(
            normalize("25℃，12.5%，3/4，2-3个。")?,
            "二十五摄氏度, 百分之十二点五, 四分之三, 二到三个."
        );
        assert!(normalize("0.12345678901234567").is_err());
        assert_eq!(normalize("价格是１２３元。")?, "价格是一百二十三元.");
        assert_eq!(normalize("2０2６年１０月２７日")?, "二零二六年十月二十七日");
        assert_eq!(
            normalize("００２６年，１２.５%，１２/３４，１２-１３个")?,
            "零零二六年, 百分之十二点五, 三十四分之十二, 十二到十三个"
        );
        assert_eq!(normalize("１２３４５６年")?, "十二三四五六年");
        assert_eq!(normalize("٢٠٢٦年，𝟚𝟘𝟚𝟞年")?, "٢٠٢٦年, 𝟚𝟘𝟚𝟞年");
        Ok(())
    }
}
