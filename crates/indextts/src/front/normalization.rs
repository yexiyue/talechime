//! Fixed WeText 0.1.0 CN/EN resources and explicit default options.
use super::{fst::FstNormalizer, reorder};
type Result<T> = std::result::Result<T, candle_core::Error>;
fn text_error(error: talechime_text::Error) -> candle_core::Error {
    candle_core::Error::Msg(error.to_string())
}
use regex::Regex;
use sha2::{Digest, Sha256};
use std::sync::OnceLock;
use std::{fs::File, io::Read, path::Path};

pub(super) const FILES: &[(&str, &str)] = &[
    (
        "zh/tn/tagger.fst",
        "63f3abeafe64fc3e9232099ab8ef6efd44c5198319193854af3ac2716d7c27a7",
    ),
    (
        "zh/tn/verbalizer.fst",
        "9fcadad76cafddb2e96b92d892d18249f0ba5fbe60e4d4fc51f466cb6bc7ade9",
    ),
    (
        "en/tn/tagger.fst",
        "bb804f88030153999f154f01950c18dbfd64de2c4cfc5035e21371d4eb4ead8a",
    ),
    (
        "en/tn/verbalizer.fst",
        "33ad65460a3e8f7fe313994ff0204a218c0fcb5dec9c09b24aeeaa30ef21d67b",
    ),
];
const LINUX_FILES: &[(&str, &str)] = &[
    (
        "zh/tagger.fst",
        "88aabfb4b0d3df506622f620538f031e3d2747083989e84aa242dec3413a9155",
    ),
    (
        "zh/verbalizer.fst",
        "6f1aa8af41f294390da58618dc8612b7e9ab198d6b8dca47764cf663bda683d5",
    ),
    (
        "en/tagger.fst",
        "d8fa27641bf5773d2cfd9486f9213697be82dbce56d8bf44f18d658a0bee33f3",
    ),
    (
        "en/verbalizer.fst",
        "d2290ffaaad4b611ccca4b480f7a70a9842785caeda6d28632f5259860ae5265",
    ),
];
const SPANISH_FILES: &[(&str, &str)] = &[
    (
        "tagger.fst",
        "fb8ecfe237d74bc19c5fbe0b6200042929118a00653c2adc755e42e260ff4f13",
    ),
    (
        "verbalizer.fst",
        "d723c5fe90e18d7b76b19357f5c75ebb7b019ae49855a5d2b35f188b608e7932",
    ),
];

fn verify_files(directory: &Path, files: &[(&str, &str)]) -> Result<()> {
    for &(name, expected) in files {
        let path = directory.join(name);
        let mut file = File::open(&path).map_err(|error| {
            candle_core::Error::Msg(format!(
                "IndexTTS requires pinned TN resource {}: {error}",
                path.display()
            ))
        })?;
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
            candle_core::bail!("IndexTTS TN resource identity mismatch: {name}");
        }
    }
    Ok(())
}

pub(super) struct Normalization {
    chinese: (FstNormalizer, FstNormalizer),
    english: (FstNormalizer, FstNormalizer),
    linux: bool,
}
impl Normalization {
    pub(super) fn load(directory: &Path) -> Result<Self> {
        Self::load_platform(directory, cfg!(target_os = "linux"))
    }
    pub(super) fn load_platform(directory: &Path, linux: bool) -> Result<Self> {
        let root = if linux {
            directory.with_file_name("linux")
        } else {
            directory.to_owned()
        };
        let files = if linux { LINUX_FILES } else { FILES };
        verify_files(&root, files)?;
        Ok(Self {
            chinese: (
                FstNormalizer::load(&root.join(files[0].0)).map_err(text_error)?,
                FstNormalizer::load(&root.join(files[1].0)).map_err(text_error)?,
            ),
            english: (
                FstNormalizer::load(&root.join(files[2].0)).map_err(text_error)?,
                FstNormalizer::load(&root.join(files[3].0)).map_err(text_error)?,
            ),
            linux,
        })
    }
    #[cfg(test)]
    pub(super) fn normalize(&mut self, text: &str, chinese: bool) -> Result<String> {
        self.normalize_with_cancel(text, chinese, &|| false)
    }
    pub(super) fn normalize_with_cancel(
        &mut self,
        text: &str,
        chinese: bool,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<String> {
        static DIGITS: OnceLock<Regex> = OnceLock::new();
        let text = text.trim();
        if !self.linux
            && !DIGITS
                .get_or_init(|| Regex::new(r"\d").expect("static regex"))
                .is_match(text)
        {
            return Ok(text.to_owned());
        }
        let (tagger, verbalizer) = if chinese {
            &self.chinese
        } else {
            &self.english
        };
        let tagged = tagger
            .apply_with_cancel(text, cancelled)
            .map_err(text_error)?;
        let ordered = if self.linux {
            reorder::reorder_linux(tagged.trim(), chinese).map_err(text_error)?
        } else {
            reorder::reorder(tagged.trim(), chinese).map_err(text_error)?
        };
        Ok(verbalizer
            .apply_with_cancel(&ordered, cancelled)
            .map_err(text_error)?
            .trim()
            .to_owned())
    }
}

pub(super) struct Spanish {
    tagger: FstNormalizer,
    verbalizer: FstNormalizer,
}
impl Spanish {
    pub(super) fn load(directory: &Path) -> Result<Self> {
        verify_files(directory, SPANISH_FILES)?;
        Ok(Self {
            tagger: FstNormalizer::load(&directory.join("tagger.fst")).map_err(text_error)?,
            verbalizer: FstNormalizer::load(&directory.join("verbalizer.fst"))
                .map_err(text_error)?,
        })
    }
    #[cfg(test)]
    pub(super) fn normalize(&self, text: &str) -> Result<String> {
        self.normalize_with_cancel(text, &|| false)
    }
    pub(super) fn normalize_with_cancel(
        &self,
        text: &str,
        cancelled: &dyn Fn() -> bool,
    ) -> Result<String> {
        let text = text.trim();
        if text.is_empty() {
            return Ok(String::new());
        }
        let tagged = self
            .tagger
            .apply_with_cancel(text, cancelled)
            .map_err(text_error)?;
        let mut outputs = Vec::new();
        for candidates in super::nemo::permutations(&tagged)? {
            let mut output = None;
            for candidate in candidates {
                if let Some(value) = self
                    .verbalizer
                    .apply_optional_with_cancel(&candidate, cancelled)
                    .map_err(text_error)?
                {
                    output = Some(value);
                    break;
                }
            }
            outputs.push(output.ok_or_else(|| {
                candle_core::Error::Msg("Spanish TN could not verbalize the tagged input".into())
            })?);
        }
        static SPACES: OnceLock<Regex> = OnceLock::new();
        Ok(SPACES
            .get_or_init(|| Regex::new(" {2,}").expect("static regex"))
            .replace_all(&outputs.join(" "), " ")
            .into_owned())
    }
}
