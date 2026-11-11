//! Fixed IndexTTS Linux/Spanish grammar resources; no runtime Python.
use crate::native_preparation::Workspace;
pub(super) use crate::native_preparation::wetext::INPUT;
use std::{
    fs::File,
    io::{Cursor, Read, Write},
};
pub(super) const GRAMMAR_OUTPUTS: &[&str] = &[
    "tn/linux/zh/tagger.fst",
    "tn/linux/zh/verbalizer.fst",
    "tn/linux/en/tagger.fst",
    "tn/linux/en/verbalizer.fst",
    "tn/es/tagger.fst",
    "tn/es/verbalizer.fst",
    "LICENSE-NEMO_TEXT_PROCESSING",
    "LICENSE-WETEXTPROCESSING",
];
pub(super) fn prepare(work: &Workspace<'_>) -> anyhow::Result<()> {
    crate::native_preparation::wetext::prepare(work)?;
    for &name in GRAMMAR_OUTPUTS {
        work.component(name, |_, output, control| {
            let mut archive =
                zip::ZipArchive::new(Cursor::new(include_bytes!("tn-grammars-v1.zip")))?;
            let mut entry = archive.by_name(name.strip_prefix("tn/").unwrap_or(name))?;
            let pin = super::assets()
                .outputs
                .iter()
                .find(|pin| pin.path == name)
                .ok_or_else(|| anyhow::anyhow!("grammar output is not pinned"))?;
            anyhow::ensure!(entry.size() == pin.size, "grammar resource size mismatch");
            let mut destination = File::create(output)?;
            let mut buffer = [0; 65536];
            loop {
                control.check()?;
                let size = entry.read(&mut buffer)?;
                if size == 0 {
                    break;
                }
                destination.write_all(&buffer[..size])?;
            }
            Ok(())
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};
    #[test]
    fn bundled_grammars_and_licenses_match_the_pinned_outputs() {
        let mut archive =
            zip::ZipArchive::new(Cursor::new(include_bytes!("tn-grammars-v1.zip"))).unwrap();
        for &name in GRAMMAR_OUTPUTS {
            let mut entry = archive
                .by_name(name.strip_prefix("tn/").unwrap_or(name))
                .unwrap();
            let pin = super::super::assets()
                .outputs
                .iter()
                .find(|pin| pin.path == name)
                .unwrap();
            let mut bytes = Vec::new();
            entry.read_to_end(&mut bytes).unwrap();
            assert_eq!(bytes.len() as u64, pin.size, "{name}");
            assert_eq!(
                format!("{:x}", Sha256::digest(&bytes)),
                pin.sha256,
                "{name}"
            );
        }
    }
}
