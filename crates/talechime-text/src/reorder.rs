//! WeText 0.1.0 TN field order and escaped values.
//! Port of wetext/token_parser.py, Copyright (c) 2022 Zhendong Peng.
//! Apache-2.0; see licenses/WETEXT-LICENSE and SOURCE.md.
use crate::Result;

fn orders(chinese: bool, name: &str) -> Option<&'static [&'static str]> {
    match (chinese, name) {
        (true, "date") => Some(&["year", "month", "day"]),
        (true, "fraction") => Some(&["denominator", "numerator"]),
        (true, "measure") => Some(&["denominator", "numerator", "value"]),
        (true, "money") => Some(&["value", "currency"]),
        (true, "time") => Some(&["noon", "hour", "minute", "second"]),
        (false, "date") => Some(&["preserve_order", "text", "day", "month", "year"]),
        (false, "money") => Some(&[
            "integer_part",
            "fractional_part",
            "quantity",
            "currency_maj",
        ]),
        _ => None,
    }
}
struct Parser<'a> {
    text: &'a str,
    position: usize,
}
impl<'a> Parser<'a> {
    fn spaces(&mut self) {
        while self.text.as_bytes().get(self.position) == Some(&b' ') {
            self.position += 1;
        }
    }
    fn take(&mut self, byte: u8) -> Result<()> {
        self.spaces();
        if self.text.as_bytes().get(self.position) != Some(&byte) {
            crate::fail!("invalid TN tagged structure at byte {}", self.position);
        }
        self.position += 1;
        Ok(())
    }
    fn key(&mut self) -> Result<&'a str> {
        self.spaces();
        let start = self.position;
        while self
            .text
            .as_bytes()
            .get(self.position)
            .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
        {
            self.position += 1;
        }
        if self.position == start {
            crate::fail!("missing TN tagged key at byte {}", start);
        }
        Ok(&self.text[start..self.position])
    }
    fn value(&mut self) -> Result<&'a str> {
        self.take(b'"')?;
        let start = self.position;
        while let Some(&byte) = self.text.as_bytes().get(self.position) {
            match byte {
                b'"' => {
                    let end = self.position;
                    self.position += 1;
                    return Ok(&self.text[start..end]);
                }
                b'\\' => self.position += 2,
                _ => self.position += 1,
            }
        }
        crate::fail!("unterminated TN tagged value");
    }
}

pub fn reorder(text: &str, chinese: bool) -> Result<String> {
    reorder_fields(text, chinese, false)
}
pub fn reorder_linux(text: &str, chinese: bool) -> Result<String> {
    reorder_fields(text, chinese, true)
}
fn reorder_fields(text: &str, chinese: bool, preserve_extra_fields: bool) -> Result<String> {
    let mut parser = Parser { text, position: 0 };
    let mut output = Vec::new();
    parser.spaces();
    while parser.position < text.len() {
        let name = parser.key()?;
        parser.take(b'{')?;
        let mut members = Vec::new();
        loop {
            parser.spaces();
            if parser.text.as_bytes().get(parser.position) == Some(&b'}') {
                parser.position += 1;
                break;
            }
            let key = parser.key()?;
            parser.take(b':')?;
            if preserve_extra_fields && members.iter().any(|(name, _)| *name == key) {
                crate::fail!("duplicate Linux TN tagged field: {key}");
            }
            members.push((key, parser.value()?));
        }
        let member = |key: &str| {
            members
                .iter()
                .rev()
                .find(|(name, _)| *name == key)
                .map(|(_, value)| *value)
        };
        let preserved = member("preserve_order") == Some("true");
        let mut order: Vec<_> = if !preserved {
            orders(chinese, name)
                .map(|order| order.to_vec())
                .unwrap_or_else(|| members.iter().map(|(key, _)| *key).collect())
        } else {
            members.iter().map(|(key, _)| *key).collect()
        };
        if preserve_extra_fields && !preserved {
            for (key, _) in &members {
                if !order.contains(key) {
                    order.push(*key);
                }
            }
        }
        let mut item = format!("{name} {{");
        for key in order {
            if let Some(value) = member(key) {
                item.push_str(&format!(" {key}: \"{value}\""));
            }
        }
        item.push_str(" }");
        output.push(item);
        parser.spaces();
    }
    Ok(output.join(" "))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dates_reorder_and_preserved_dates_do_not() {
        assert_eq!(
            reorder(
                "date { day: \"二\" year: \"二零二六\" month: \"一\" }",
                true
            )
            .unwrap(),
            "date { year: \"二零二六\" month: \"一\" day: \"二\" }"
        );
        let original = "date { month: \"May\" day: \"first\" preserve_order: \"true\" }";
        assert_eq!(reorder(original, false).unwrap(), original);
    }
    #[test]
    fn escaped_values_survive_and_malformed_output_fails() {
        let original = r#"name { text: "a\"b\\c" }"#;
        assert_eq!(reorder(original, true).unwrap(), original);
        assert!(reorder("name { text: \"unterminated", true).is_err());
        assert!(reorder("name { text: 123 }", true).is_err());
    }
}
