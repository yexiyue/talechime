//! Fixed WeText 0.1.0 TN, with optional transformations disabled.
use crate::{Error, FstNormalizer, Result, reorder};
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path, sync::OnceLock};

/// Relative path, size and SHA-256 of the fixed wheel resources.
pub const FILES: &[(&str, u64, &str)] = &[
    (
        "zh/tn/tagger.fst",
        811646,
        "63f3abeafe64fc3e9232099ab8ef6efd44c5198319193854af3ac2716d7c27a7",
    ),
    (
        "zh/tn/verbalizer.fst",
        87702,
        "9fcadad76cafddb2e96b92d892d18249f0ba5fbe60e4d4fc51f466cb6bc7ade9",
    ),
    (
        "en/tn/tagger.fst",
        5220942,
        "bb804f88030153999f154f01950c18dbfd64de2c4cfc5035e21371d4eb4ead8a",
    ),
    (
        "en/tn/verbalizer.fst",
        1822714,
        "33ad65460a3e8f7fe313994ff0204a218c0fcb5dec9c09b24aeeaa30ef21d67b",
    ),
];

/// A fixed-resource executor; callers choose if/when to apply it.
pub struct WeText {
    chinese: (FstNormalizer, FstNormalizer),
    english: (FstNormalizer, FstNormalizer),
}
impl WeText {
    /// Reject missing or modified resources before loading the graphs.
    pub fn load(directory: &Path) -> Result<Self> {
        for &(name, _, expected) in FILES {
            let mut file = File::open(directory.join(name))?;
            let mut hash = Sha256::new();
            let mut buffer = [0; 65536];
            loop {
                let size = file.read(&mut buffer)?;
                if size == 0 {
                    break;
                }
                hash.update(&buffer[..size]);
            }
            if format!("{:x}", hash.finalize()) != expected {
                return Err(Error::Invalid(format!(
                    "WeText resource identity mismatch: {name}"
                )));
            }
        }
        Ok(Self {
            chinese: (
                FstNormalizer::load(&directory.join(FILES[0].0))?,
                FstNormalizer::load(&directory.join(FILES[1].0))?,
            ),
            english: (
                FstNormalizer::load(&directory.join(FILES[2].0))?,
                FstNormalizer::load(&directory.join(FILES[3].0))?,
            ),
        })
    }
    /// Default auto language and should_normalize behavior of the fixed wheel.
    pub fn normalize(&self, text: &str) -> Result<String> {
        self.normalize_with_cancel(text, &|| false)
    }
    /// Normalize with cancellation between graph stages and path-search steps.
    pub fn normalize_with_cancel(
        &self,
        text: &str,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<String> {
        if cancelled() {
            return Err(Error::Cancelled);
        }
        let text = text.trim();
        static DIGITS: OnceLock<regex::Regex> = OnceLock::new();
        if !DIGITS
            .get_or_init(|| regex::Regex::new(r"\d").expect("static regex"))
            .is_match(text)
        {
            return Ok(text.to_owned());
        }
        let chinese = text
            .chars()
            .any(|ch| ('\u{4e00}'..='\u{9fff}').contains(&ch))
            || text.chars().all(|ch| ch.is_numeric());
        let (tagger, verbalizer) = if chinese {
            &self.chinese
        } else {
            &self.english
        };
        let tagged = tagger.apply_with_cancel(text, cancelled)?;
        verbalizer
            .apply_with_cancel(&reorder(&tagged, chinese)?, cancelled)
            .map(|text| text.trim().to_owned())
    }
}
