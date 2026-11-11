//! Lossless conversion of the fixed unidic-lite MeCab tables to Vibrato.
//! The binary layout follows MeCab 0.996; no C++ or Python runs in the product.
use crate::native_preparation::PreparationControl;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufWriter, Read, Write},
    path::Path,
};

pub(super) const INPUT: crate::native_preparation::Input = crate::native_preparation::Input {
    path: "unidic-lite-1.0.8.tar.gz",
    repo: "polm/unidic-lite",
    revision: "1.0.8",
    filename: "unidic-lite-1.0.8.tar.gz",
    size: 47356746,
    sha256: "db9d4572d9fdd4d00a97949d4b0741ec480ee05a7e7e2e32f547500dae27b245",
    url: Some(
        "https://files.pythonhosted.org/packages/55/2b/8cf7514cb57d028abcef625afa847d60ff1ffbf0049c36b78faa7c35046f/unidic-lite-1.0.8.tar.gz",
    ),
};
const FILES: &[(&str, &str)] = &[
    (
        "sys.dic",
        "122c4c91f026bf4b65bbe2dfd8b9f8eeb6ab56d8eb2a7287a68bca75708ba513",
    ),
    (
        "unk.dic",
        "e3b92803feeb6c2712c796b905317ec192a59729d2c1e92b2728d454169541c4",
    ),
    (
        "char.bin",
        "dd31396563d8924645b80fd3c9aa7b13ca089d7748f25553a1d6bc3f9b511ae8",
    ),
    (
        "matrix.bin",
        "88fb99efe1075d9b8b40e01bb2751e4095f1612618f2490af776eb5ed39190a9",
    ),
    (
        "BSD",
        "7917ecfaf95a5ba7b4f45aa9a845ac931d79c90f39b7e711442dbc21c612560f",
    ),
];

fn u32_at(data: &[u8], offset: usize) -> anyhow::Result<u32> {
    let bytes = data
        .get(offset..offset + 4)
        .ok_or_else(|| anyhow::anyhow!("truncated MeCab table"))?;
    Ok(u32::from_le_bytes(bytes.try_into()?))
}
fn u16_at(data: &[u8], offset: usize) -> anyhow::Result<u16> {
    let bytes = data
        .get(offset..offset + 2)
        .ok_or_else(|| anyhow::anyhow!("truncated MeCab token"))?;
    Ok(u16::from_le_bytes(bytes.try_into()?))
}
struct Lexicon<'a> {
    data: &'a [u8],
    trie_size: usize,
    token_size: usize,
    entries: usize,
}
impl<'a> Lexicon<'a> {
    fn new(data: &'a [u8], kind: u32) -> anyhow::Result<Self> {
        anyhow::ensure!(
            data.len() >= 72 && u32_at(data, 4)? == 102 && u32_at(data, 8)? == kind,
            "unexpected MeCab dictionary header"
        );
        let trie_size = u32_at(data, 24)? as usize;
        let token_size = u32_at(data, 28)? as usize;
        let feature_size = u32_at(data, 32)? as usize;
        let entries = u32_at(data, 12)? as usize;
        anyhow::ensure!(
            trie_size.is_multiple_of(8)
                && token_size == entries * 16
                && 72 + trie_size + token_size + feature_size == data.len(),
            "invalid MeCab dictionary coverage"
        );
        anyhow::ensure!(
            data[40..72].starts_with(b"utf8\0") || data[40..72].starts_with(b"UTF-8\0"),
            "MeCab dictionary must be UTF-8"
        );
        Ok(Self {
            data,
            trie_size,
            token_size,
            entries,
        })
    }
    fn unit(&self, index: usize) -> anyhow::Result<(i32, u32)> {
        anyhow::ensure!(index < self.trie_size / 8, "MeCab trie index out of bounds");
        Ok((
            u32_at(self.data, 72 + index * 8)? as i32,
            u32_at(self.data, 76 + index * 8)?,
        ))
    }
    fn dump(&self, writer: &mut impl Write, control: &PreparationControl) -> anyhow::Result<()> {
        let mut seen = vec![false; self.entries];
        self.visit(
            self.unit(0)?.0 as usize,
            &mut Vec::new(),
            &mut seen,
            writer,
            control,
        )?;
        anyhow::ensure!(
            seen.into_iter().all(|entry| entry),
            "MeCab conversion omitted lexical entries"
        );
        Ok(())
    }
    fn visit(
        &self,
        base: usize,
        surface: &mut Vec<u8>,
        seen: &mut [bool],
        writer: &mut impl Write,
        control: &PreparationControl,
    ) -> anyhow::Result<()> {
        control.check()?;
        anyhow::ensure!(
            surface.len() <= 1024,
            "MeCab trie has a cycle or oversized surface"
        );
        let (value, check) = self.unit(base)?;
        if check as usize == base && value < 0 {
            let encoded = -(value as i64) - 1;
            let start = (encoded >> 8) as usize;
            let count = (encoded & 255) as usize;
            anyhow::ensure!(
                start + count <= self.entries,
                "MeCab token range out of bounds"
            );
            let surface = std::str::from_utf8(surface)?;
            for (index, visited) in seen.iter_mut().enumerate().skip(start).take(count) {
                anyhow::ensure!(!*visited, "duplicate MeCab token range");
                *visited = true;
                let offset = 72 + self.trie_size + index * 16;
                let left = u16_at(self.data, offset)?;
                let right = u16_at(self.data, offset + 2)?;
                let cost = u16_at(self.data, offset + 6)? as i16;
                let feature = u32_at(self.data, offset + 8)? as usize;
                let features = self
                    .data
                    .get(72 + self.trie_size + self.token_size + feature..)
                    .ok_or_else(|| anyhow::anyhow!("MeCab feature offset out of bounds"))?;
                let end = features
                    .iter()
                    .position(|byte| *byte == 0)
                    .ok_or_else(|| anyhow::anyhow!("unterminated MeCab feature"))?;
                writeln!(
                    writer,
                    "\"{}\",{left},{right},{cost},{}",
                    surface.replace('"', "\"\""),
                    std::str::from_utf8(&features[..end])?
                )?;
            }
        }
        for byte in 0..=255u8 {
            let index = base + usize::from(byte) + 1;
            if index >= self.trie_size / 8 {
                continue;
            }
            let (next, check) = self.unit(index)?;
            if check as usize == base && next >= 0 {
                surface.push(byte);
                self.visit(next as usize, surface, seen, writer, control)?;
                surface.pop();
            }
        }
        Ok(())
    }
}

fn char_def(data: &[u8]) -> anyhow::Result<Vec<u8>> {
    let count = u32_at(data, 0)? as usize;
    anyhow::ensure!(
        count > 0 && count <= 18 && data.len() == 4 + count * 32 + 65535 * 4,
        "invalid MeCab character table"
    );
    let mut names = Vec::new();
    for index in 0..count {
        let name = &data[4 + index * 32..4 + (index + 1) * 32];
        let end = name
            .iter()
            .position(|byte| *byte == 0)
            .ok_or_else(|| anyhow::anyhow!("invalid MeCab category name"))?;
        names.push(std::str::from_utf8(&name[..end])?);
    }
    let mut values: Vec<_> = (0..65535)
        .map(|i| u32_at(data, 4 + count * 32 + i * 4))
        .collect::<anyhow::Result<_>>()?;
    // MeCab stores 0xffff entries. Vibrato covers the entire BMP, with the
    // otherwise unspecified U+FFFF retaining the source char.def DEFAULT.
    values.push(values[0]);
    let mut flags = vec![None; count];
    for &value in &values {
        let base = ((value >> 18) & 255) as usize;
        anyhow::ensure!(
            base < count && value & ((1 << count) - 1) != 0,
            "invalid MeCab character category"
        );
        let properties = value >> 26;
        if let Some(previous) = flags[base] {
            anyhow::ensure!(
                previous == properties,
                "inconsistent MeCab category properties"
            );
        }
        flags[base] = Some(properties);
    }
    let mut output = Vec::new();
    for (name, flag) in names.iter().zip(flags) {
        let flag = flag.ok_or_else(|| anyhow::anyhow!("unused MeCab category"))?;
        writeln!(
            output,
            "{name} {} {} {}",
            (flag >> 5) & 1,
            (flag >> 4) & 1,
            flag & 15
        )?;
    }
    let mut start = 0;
    while start < values.len() {
        let value = values[start];
        let mut end = start + 1;
        while end < values.len() && values[end] == value {
            end += 1;
        }
        let base = ((value >> 18) & 255) as usize;
        write!(output, "0x{start:04X}..0x{:04X} {}", end - 1, names[base])?;
        for (index, name) in names.iter().enumerate() {
            if index != base && value & (1 << index) != 0 {
                write!(output, " {name}")?;
            }
        }
        writeln!(output)?;
        start = end;
    }
    Ok(output)
}

fn matrix_def(data: &[u8], path: &Path, control: &PreparationControl) -> anyhow::Result<()> {
    let right = u16_at(data, 0)? as usize;
    let left = u16_at(data, 2)? as usize;
    anyhow::ensure!(
        right > 0 && left > 0 && data.len() == 4 + right * left * 2,
        "invalid MeCab cost matrix"
    );
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(writer, "{right} {left}")?;
    for r in 0..right {
        control.check()?;
        for l in 0..left {
            writeln!(
                writer,
                "{r} {l} {}",
                u16_at(data, 4 + (r + l * right) * 2)? as i16
            )?;
        }
    }
    writer.flush()?;
    Ok(())
}

pub(super) fn convert(
    input: &Path,
    output: &Path,
    control: &PreparationControl,
) -> anyhow::Result<()> {
    let workspace = tempfile::tempdir_in(
        output
            .parent()
            .ok_or_else(|| anyhow::anyhow!("missing output parent"))?,
    )?;
    let decoder = flate2::read::GzDecoder::new(File::open(input.join(INPUT.path))?);
    let mut archive = tar::Archive::new(decoder);
    let mut files = std::collections::BTreeMap::new();
    for entry in archive.entries()? {
        control.check()?;
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        for &(name, hash) in FILES {
            if path == Path::new("unidic-lite-1.0.8/unidic_lite/dicdir").join(name) {
                anyhow::ensure!(
                    entry.size() <= 200_000_000 && !files.contains_key(name),
                    "invalid dictionary archive entry"
                );
                let mut bytes = Vec::new();
                entry.read_to_end(&mut bytes)?;
                anyhow::ensure!(
                    format!("{:x}", Sha256::digest(&bytes)) == hash,
                    "unidic-lite component identity mismatch: {name}"
                );
                files.insert(name, bytes);
            }
        }
    }
    anyhow::ensure!(
        files.len() == FILES.len(),
        "unidic-lite archive is incomplete"
    );
    for (name, kind, target) in [("sys.dic", 0, "lex.csv"), ("unk.dic", 2, "unk.def")] {
        let mut writer = BufWriter::new(File::create(workspace.path().join(target))?);
        Lexicon::new(&files[name], kind)?.dump(&mut writer, control)?;
        writer.flush()?;
    }
    let characters = char_def(&files["char.bin"])?;
    matrix_def(
        &files["matrix.bin"],
        &workspace.path().join("matrix.def"),
        control,
    )?;
    drop(files);
    control.check()?;
    let dictionary = vibrato::SystemDictionaryBuilder::from_readers(
        File::open(workspace.path().join("lex.csv"))?,
        File::open(workspace.path().join("matrix.def"))?,
        characters.as_slice(),
        File::open(workspace.path().join("unk.def"))?,
    )?;
    control.check()?;
    dictionary.write(BufWriter::new(File::create(output)?))?;
    control.check()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn corrupt_binary_tables_return_errors() {
        assert!(Lexicon::new(&[], 0).is_err());
        assert!(char_def(&[0; 32]).is_err());
        let root = tempfile::tempdir().unwrap();
        assert!(
            matrix_def(
                &[0; 8],
                &root.path().join("matrix"),
                &PreparationControl::default()
            )
            .is_err()
        );
    }
    #[test]
    #[ignore = "explicit fixed unidic-lite archive and new output directory"]
    fn fixed_binary_dictionary_converts_to_the_pinned_vibrato_identity() {
        let input = std::path::PathBuf::from(
            std::env::var_os("TALECHIME_INDEXTTS_JA_INPUTS").expect("input directory"),
        );
        let output = std::path::PathBuf::from(
            std::env::var_os("TALECHIME_INDEXTTS_JA_OUTPUT").expect("new output path"),
        );
        assert!(!output.exists());
        convert(&input, &output, &PreparationControl::default()).unwrap();
        let data = std::fs::read(output).unwrap();
        let pin = super::super::assets()
            .outputs
            .iter()
            .find(|pin| pin.path == "japanese.dic")
            .unwrap();
        assert_eq!(data.len() as u64, pin.size);
        assert_eq!(format!("{:x}", Sha256::digest(data)), pin.sha256);
    }
}
