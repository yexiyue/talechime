//! The misaki 0.9.4 lexicons, exactly as hexgrad ships them (Apache-2.0).
//!
//! Kokoro was trained on phonemes from these files, so they define the
//! alphabet its voices expect: `O` for the American GOAT vowel, rhotic `ɜɹ`,
//! and part-of-speech variants for heteronyms (`record` as noun and verb).

use std::collections::HashMap;

use crate::en::{Entry, Lexicon};
use crate::{Dialect, Error};

pub(crate) fn load(dialect: Dialect) -> Result<Lexicon, Error> {
    let (gold, silver) = match dialect {
        Dialect::American => (include_str!("../data/misaki-us_gold.json"), include_str!("../data/misaki-us_silver.json")),
        Dialect::British => (include_str!("../data/misaki-gb_gold.json"), include_str!("../data/misaki-gb_silver.json")),
    };
    let parse = |json: &str| serde_json::from_str::<HashMap<String, Entry>>(json).map_err(|e| Error::Weights(format!("lexicon: {e}")));
    Ok(Lexicon::new(dialect == Dialect::British, parse(gold)?, parse(silver)?))
}
