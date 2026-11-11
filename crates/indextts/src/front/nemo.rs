//! NeMo 1.2.0 token structures and bounded field-order permutations.
//! Apache-2.0, NVIDIA; see licenses/NEMO_TEXT_PROCESSING and SOURCE.md.
use candle_core::Result;

#[derive(Clone, Debug)]
enum Value {
    Text(String),
    Block(Vec<Field>),
    Preserve,
}
#[derive(Clone, Debug)]
struct Field {
    key: String,
    value: Value,
}
struct Parser<'a> {
    text: &'a [u8],
    position: usize,
}
impl Parser<'_> {
    fn spaces(&mut self) {
        while self.text.get(self.position) == Some(&b' ') {
            self.position += 1;
        }
    }
    fn fields(&mut self, nested: bool) -> Result<Vec<Field>> {
        let mut fields = Vec::new();
        loop {
            self.spaces();
            match self.text.get(self.position) {
                None if !nested => return Ok(fields),
                Some(b'}') if nested => {
                    self.position += 1;
                    return Ok(fields);
                }
                _ => {}
            }
            let start = self.position;
            while self
                .text
                .get(self.position)
                .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
            {
                self.position += 1;
            }
            if start == self.position {
                candle_core::bail!("invalid NeMo tagged key");
            }
            let key = String::from_utf8(self.text[start..self.position].to_vec())
                .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
            self.spaces();
            let value = match self.text.get(self.position) {
                Some(b'{') => {
                    self.position += 1;
                    Value::Block(self.fields(true)?)
                }
                Some(b':') => {
                    self.position += 1;
                    self.spaces();
                    if key == "preserve_order" && self.text[self.position..].starts_with(b"true") {
                        self.position += 4;
                        Value::Preserve
                    } else {
                        if self.text.get(self.position) != Some(&b'"') {
                            candle_core::bail!("invalid NeMo tagged value");
                        }
                        self.position += 1;
                        let start = self.position;
                        // NeMo values end at quote + space, and do not unescape
                        // their content (including embedded quotes/backslashes).
                        while self.position + 1 < self.text.len()
                            && !(self.text[self.position] == b'"'
                                && self.text[self.position + 1] == b' ')
                        {
                            self.position += 1;
                        }
                        if self.position + 1 >= self.text.len() {
                            candle_core::bail!("unterminated NeMo tagged value");
                        }
                        let value = std::str::from_utf8(&self.text[start..self.position])
                            .map_err(|error| candle_core::Error::Msg(error.to_string()))?
                            .to_owned();
                        self.position += 1;
                        Value::Text(value)
                    }
                }
                _ => candle_core::bail!("invalid NeMo tagged structure"),
            };
            // Python OrderedDict overwrites duplicate keys without moving them.
            if let Some(field) = fields
                .iter_mut()
                .find(|field: &&mut Field| nested && field.key == key)
            {
                field.value = value;
            } else {
                fields.push(Field { key, value });
            }
        }
    }
}

const LIMIT: usize = 729;
fn multiply(left: usize, right: usize) -> Result<usize> {
    let value = left
        .checked_mul(right)
        .ok_or_else(|| candle_core::Error::Msg("NeMo permutation budget overflow".into()))?;
    if value > LIMIT {
        candle_core::bail!("NeMo token exceeds the fixed 729 permutation limit");
    }
    Ok(value)
}
fn count(fields: &[Field]) -> Result<usize> {
    let mut total = 1;
    if !fields.iter().any(|field| field.key == "preserve_order") {
        for factor in 2..=fields.len() {
            total = multiply(total, factor)?;
        }
    }
    for field in fields {
        if let Value::Block(children) = &field.value {
            total = multiply(total, count(children)?)?;
        }
    }
    Ok(total)
}

fn orders(fields: &[Field]) -> Vec<Vec<usize>> {
    fn append(prefix: &mut Vec<usize>, used: &mut [bool], output: &mut Vec<Vec<usize>>) {
        if prefix.len() == used.len() {
            output.push(prefix.clone());
            return;
        }
        for index in 0..used.len() {
            if !used[index] {
                used[index] = true;
                prefix.push(index);
                append(prefix, used, output);
                prefix.pop();
                used[index] = false;
            }
        }
    }
    if fields.iter().any(|field| field.key == "preserve_order") {
        return vec![(0..fields.len()).collect()];
    }
    let mut output = Vec::new();
    append(&mut Vec::new(), &mut vec![false; fields.len()], &mut output);
    output
}
fn serializations(fields: &[Field]) -> Vec<String> {
    let mut output = Vec::new();
    for order in orders(fields) {
        let mut prefixes = vec![String::new()];
        for index in order {
            let field = &fields[index];
            let variants = match &field.value {
                Value::Text(value) => vec![format!("{}: \"{value}\" ", field.key)],
                Value::Preserve => vec![format!("{}: true ", field.key)],
                Value::Block(children) => serializations(children)
                    .into_iter()
                    .map(|text| format!(" {} {{ {text} }} ", field.key))
                    .collect(),
            };
            prefixes = prefixes
                .iter()
                .flat_map(|prefix| {
                    variants
                        .iter()
                        .map(move |suffix| format!("{prefix}{suffix}"))
                })
                .collect();
        }
        output.extend(prefixes);
    }
    output
}

/// Split token groups before the Cartesian product exceeds the upstream budget;
/// return permutations in Python itertools order, including preserve_order.
pub(super) fn permutations(text: &str) -> Result<Vec<Vec<String>>> {
    let tokens = Parser {
        text: text.as_bytes(),
        position: 0,
    }
    .fields(false)?;
    let mut output = Vec::new();
    let mut group = vec![String::new()];
    let mut size = 1;
    for token in tokens {
        let number = count(std::slice::from_ref(&token))?;
        if size * number > LIMIT {
            output.push(group);
            group = vec![String::new()];
            size = 1;
        }
        size = multiply(size, number)?;
        let variants = serializations(&[token]);
        group = group
            .iter()
            .flat_map(|prefix| {
                variants
                    .iter()
                    .map(move |suffix| format!("{prefix}{suffix}"))
            })
            .collect();
    }
    if size > 1 || group != [String::new()] {
        output.push(group);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preserve_order_and_nested_fields_match_python_serialization() {
        assert_eq!(
            permutations("tokens { date { month: \"mayo\" day: \"uno\" preserve_order: true } } ")
                .unwrap(),
            vec![vec![
                " tokens {  date { month: \"mayo\" day: \"uno\" preserve_order: true  }  } "
                    .to_owned()
            ]]
        );
        assert_eq!(
            permutations("tokens { cardinal { integer: \"dos\" negative: \"true\" } } ").unwrap()
                [0]
            .len(),
            2
        );
        assert!(permutations("tokens { name: \"broken").is_err());
        assert!(
            permutations("tokens { name: \"Hola\" } tokens { name: \"mundo\" } ").unwrap()[0][0]
                .contains("Hola")
        );
        assert!(
            permutations("tokens { name: \"Hola\" } tokens { name: \"mundo\" } ").unwrap()[0][0]
                .contains("mundo")
        );
    }
}
