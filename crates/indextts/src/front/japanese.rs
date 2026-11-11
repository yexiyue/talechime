//! Fixed ratio=0 UniDic surface segmentation; Japanese TN is a passthrough.
use std::{fs::File, io::Read, path::Path};

use candle_core::{Result, bail};
use sha2::{Digest, Sha256};

/// Conversion of the exact unidic-lite 1.0.8 lexical/cost tables with Vibrato 0.5.3.
pub(crate) const DICTIONARY_SHA256: &str =
    "74f2526bcb8637f387bb688ecd15aced3001b349275e7ec921c68324ebec2ddd";

pub(crate) struct JapaneseFrontend {
    tokenizer: vibrato::Tokenizer,
}

impl JapaneseFrontend {
    pub(crate) fn load(path: &Path) -> Result<Self> {
        let mut file = File::open(path).map_err(|error| {
            candle_core::Error::Msg(format!(
                "IndexTTS Japanese requires the pinned UniDic resource {}: {error}",
                path.display()
            ))
        })?;
        let mut hash = Sha256::new();
        let mut buffer = [0; 64 * 1024];
        loop {
            let size = file.read(&mut buffer)?;
            if size == 0 {
                break;
            }
            hash.update(&buffer[..size]);
        }
        if format!("{:x}", hash.finalize()) != DICTIONARY_SHA256 {
            bail!("IndexTTS Japanese dictionary identity mismatch; prepare the fixed resource");
        }
        let dictionary = vibrato::Dictionary::read(File::open(path)?)
            .map_err(|error| candle_core::Error::Msg(format!("Japanese dictionary: {error}")))?;
        Self::from_dictionary(dictionary)
    }

    fn from_dictionary(dictionary: vibrato::Dictionary) -> Result<Self> {
        let tokenizer = vibrato::Tokenizer::new(dictionary)
            .ignore_space(true)
            .map_err(|error| candle_core::Error::Msg(format!("Japanese dictionary: {error}")))?
            .max_grouping_len(24);
        Ok(Self { tokenizer })
    }

    /// Match `re.split(r'( +)', text)` and ratio=0 surface joining. Only ASCII
    /// spaces delimit segments; other whitespace is handled by MeCab's table.
    #[cfg(test)]
    pub(crate) fn process(&self, text: &str) -> String {
        self.process_with_cancel(text, &|| false).unwrap()
    }

    pub(crate) fn process_with_cancel(
        &self,
        text: &str,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<String> {
        let mut worker = self.tokenizer.new_worker();
        let mut output = String::with_capacity(text.len());
        for part in text.split_inclusive(' ') {
            if cancelled() {
                bail!("Japanese text processing cancelled");
            }
            let segment = part.trim_end_matches(' ');
            if segment
                .chars()
                .all(|ch| ch.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&ch))
            {
                output.push_str(segment);
            } else {
                worker.reset_sentence(segment);
                worker.tokenize();
                for index in 0..worker.num_tokens() {
                    if index > 0 {
                        output.push(' ');
                    }
                    output.push_str(worker.token(index).surface());
                }
            }
            output.push_str(&part[segment.len()..]);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_frontend() -> JapaneseFrontend {
        let dictionary = vibrato::SystemDictionaryBuilder::from_readers(
            "東京,0,0,-100\n年,0,0,-100\n".as_bytes(),
            "1 1\n0 0 0\n".as_bytes(),
            "DEFAULT 1 1 0\nSPACE 0 1 0\nNUMERIC 1 1 0\n0x0020 SPACE\n0x0009 SPACE\n0x0030..0x0039 NUMERIC\n".as_bytes(),
            "DEFAULT,0,0,100,*\nSPACE,0,0,100,*\nNUMERIC,0,0,0,*\n".as_bytes(),
        )
        .unwrap();
        JapaneseFrontend::from_dictionary(dictionary).unwrap()
    }

    #[test]
    fn ascii_space_runs_and_blank_segments_are_preserved() {
        let frontend = small_frontend();
        assert_eq!(frontend.process("  東京   東京 "), "  東京   東京 ");
        assert_eq!(frontend.process("\t"), "\t");
        assert_eq!(frontend.process("\u{1c}"), "\u{1c}");
        assert_eq!(frontend.process(""), "");
    }

    #[test]
    fn japanese_numbers_are_not_chinese_normalized() {
        let frontend = small_frontend();
        assert_eq!(frontend.process("2026年東京"), "2026 年 東京");
        assert_eq!(frontend.process("zorbulate🙂"), "zorbulate🙂");
    }

    #[test]
    fn missing_dictionary_is_an_explicit_error() {
        let path = Path::new("missing-fixed-japanese-dictionary.dic");
        assert!(JapaneseFrontend::load(path).is_err());
    }

    #[test]
    fn corrupt_or_different_dictionary_is_rejected_before_tokenization() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("japanese.dic");
        std::fs::write(&path, b"a different dictionary version").unwrap();
        let error = JapaneseFrontend::load(&path).err().unwrap().to_string();
        assert!(error.contains("identity mismatch"));
    }

    #[test]
    #[ignore = "requires explicit locally converted unidic-lite 1.0.8 dictionary"]
    fn fixed_dictionary_matches_independent_official_oracle() {
        let path = std::env::var_os("TALECHIME_INDEXTTS_JAPANESE_DICTIONARY")
            .expect("set the exact converted dictionary path");
        let frontend = JapaneseFrontend::load(Path::new(&path)).unwrap();
        let oracle: serde_json::Value =
            serde_json::from_str(include_str!("../../tests/fixtures/japanese-oracle.json"))
                .unwrap();
        for case in oracle["cases"].as_array().unwrap() {
            let text = case["text"].as_str().unwrap();
            assert_eq!(
                frontend.process(text),
                case["output"].as_str().unwrap(),
                "{text:?}"
            );
            let prepared = super::super::TextFrontend::new()
                .prepare(text, "ja", true, Some(&frontend))
                .unwrap();
            assert_eq!(
                prepared,
                case["prepared"].as_str().unwrap(),
                "full frontend: {text:?}"
            );
        }
    }
}
