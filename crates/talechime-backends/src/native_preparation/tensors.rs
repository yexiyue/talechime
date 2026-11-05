//! Tensor-only import from fixed, SHA-verified PyTorch ZIP checkpoints.
use super::{PreparationControl, plan};
#[cfg(any(test, feature = "cosyvoice3", feature = "indextts"))]
use candle_core::pickle::{Object, Stack, TensorInfo};
use candle_core::{Device, Tensor};
#[cfg(any(test, feature = "cosyvoice3", feature = "indextts"))]
use std::io::BufReader;
use std::{
    collections::BTreeMap,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

const MAX_METADATA: u64 = 8 * 1024 * 1024;
const MAX_ENTRIES: usize = 100_000;
// IndexTTS's approved F32 text embedding alone is 309,806,080 bytes.
const MAX_TENSOR_BYTES: usize = 512 * 1024 * 1024;

#[cfg(any(test, feature = "cosyvoice3", feature = "indextts"))]
pub(crate) struct Archive {
    path: PathBuf,
    tensors: BTreeMap<String, TensorInfo>,
}
#[cfg(any(test, feature = "cosyvoice3", feature = "indextts"))]
impl Archive {
    pub fn open(
        path: &Path,
        size: u64,
        sha256: &str,
        selection: &[&str],
        control: &PreparationControl,
    ) -> anyhow::Result<Self> {
        anyhow::ensure!(
            plan::verified(path, size, sha256, control)?,
            "unverified checkpoint input"
        );
        let mut zip = zip::ZipArchive::new(File::open(path)?)?;
        anyhow::ensure!(zip.len() <= MAX_ENTRIES, "too many checkpoint entries");
        let names: Vec<String> = zip
            .file_names()
            .filter(|name| name.ends_with("/data.pkl"))
            .map(str::to_owned)
            .collect();
        anyhow::ensure!(
            names.len() == 1,
            "checkpoint requires exactly one metadata record"
        );
        let name = &names[0];
        anyhow::ensure!(
            plan::safe_relative(Path::new(name)),
            "unsafe checkpoint metadata path"
        );
        let mut metadata = zip.by_name(name)?;
        anyhow::ensure!(
            metadata.size() <= MAX_METADATA,
            "checkpoint metadata exceeds budget"
        );
        let mut bytes = Vec::new();
        (&mut metadata)
            .take(MAX_METADATA + 1)
            .read_to_end(&mut bytes)?;
        anyhow::ensure!(
            bytes.len() as u64 <= MAX_METADATA,
            "checkpoint metadata exceeds budget"
        );
        drop(metadata);
        let mut stack = Stack::empty();
        stack.read_loop(&mut BufReader::new(bytes.as_slice()))?;
        let mut value = stack.finalize()?;
        for key in selection {
            value = select(value, key)?;
        }
        let prefix = Path::new(
            name.strip_suffix(".pkl")
                .ok_or_else(|| anyhow::anyhow!("invalid checkpoint metadata name"))?,
        );
        let mut tensors = BTreeMap::new();
        flatten(value, "", prefix, &mut tensors, 0)?;
        anyhow::ensure!(
            !tensors.is_empty() && tensors.len() <= MAX_ENTRIES,
            "invalid checkpoint tensor count"
        );
        for info in tensors.values() {
            control.check()?;
            validate(info)?;
            anyhow::ensure!(
                plan::safe_relative(Path::new(&info.path))
                    && Path::new(&info.path).starts_with(prefix),
                "unsafe tensor storage path"
            );
            let storage = zip.by_name(&info.path)?;
            let storage_bytes = info
                .storage_size
                .checked_mul(info.dtype.size_in_bytes())
                .ok_or_else(|| anyhow::anyhow!("storage size overflow"))?;
            anyhow::ensure!(
                storage.size() == storage_bytes as u64,
                "tensor storage size mismatch"
            );
        }
        Ok(Self {
            path: path.into(),
            tensors,
        })
    }
    pub fn tensors(&self) -> &BTreeMap<String, TensorInfo> {
        &self.tensors
    }
    pub fn get(&self, name: &str, control: &PreparationControl) -> anyhow::Result<Tensor> {
        control.check()?;
        let info = self
            .tensors
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("checkpoint tensor missing: {name}"))?;
        validate(info)?;
        let mut zip = zip::ZipArchive::new(File::open(&self.path)?)?;
        let mut storage = zip.by_name(&info.path)?;
        let offset = info
            .layout
            .start_offset()
            .checked_mul(info.dtype.size_in_bytes())
            .ok_or_else(|| anyhow::anyhow!("tensor byte offset overflow"))?;
        // Storage offsets are in elements, not bytes. Avoid allocating preceding storage.
        let skipped = std::io::copy(
            &mut (&mut storage).take(offset as u64),
            &mut std::io::sink(),
        )?;
        anyhow::ensure!(skipped == offset as u64, "truncated tensor offset");
        control.check()?;
        let byte_count = info.layout.dims().iter().product::<usize>() * info.dtype.size_in_bytes();
        let mut bytes = Vec::new();
        bytes.try_reserve_exact(byte_count)?;
        bytes.resize(byte_count, 0);
        for chunk in bytes.chunks_mut(64 * 1024) {
            control.check()?;
            storage.read_exact(chunk)?;
        }
        let tensor = Tensor::from_raw_buffer(&bytes, info.dtype, info.layout.dims(), &Device::Cpu)?;
        let rank = info.layout.shape().rank();
        if rank > 1 && info.layout.is_fortran_contiguous() && !info.layout.is_contiguous() {
            let shape: Vec<_> = info.layout.dims().iter().rev().copied().collect();
            let axes: Vec<_> = (0..rank).rev().collect();
            Ok(tensor
                .reshape(shape)?
                .permute(axes)?
                .to_device(&Device::Cpu)?)
        } else {
            Ok(tensor)
        }
    }
}
#[cfg(any(test, feature = "cosyvoice3", feature = "indextts"))]
fn select(value: Object, key: &str) -> anyhow::Result<Object> {
    let Object::Dict(items) = value else {
        anyhow::bail!("checkpoint selection is not a mapping")
    };
    items
        .into_iter()
        .find_map(|(name, value)| (name == Object::Unicode(key.into())).then_some(value))
        .ok_or_else(|| anyhow::anyhow!("checkpoint selection missing: {key}"))
}
#[cfg(any(test, feature = "cosyvoice3", feature = "indextts"))]
fn flatten(
    value: Object,
    name: &str,
    prefix: &Path,
    tensors: &mut BTreeMap<String, TensorInfo>,
    depth: usize,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        depth <= 8 && name.len() <= 1024 && tensors.len() <= MAX_ENTRIES,
        "checkpoint nesting exceeds budget"
    );
    match value {
        Object::Dict(items) => {
            for (key, value) in items {
                let Object::Unicode(key) = key else {
                    anyhow::bail!("non-string tensor key")
                };
                if key == "_metadata" {
                    validate_metadata(value, 0)?;
                    continue;
                }
                let key = if name.is_empty() {
                    key
                } else {
                    format!("{name}.{key}")
                };
                flatten(value, &key, prefix, tensors, depth + 1)?;
            }
        }
        value => {
            let tensor_name = if name.is_empty() { "value" } else { name };
            let mut info = value
                .into_tensor_info(Object::Unicode(tensor_name.into()), prefix)?
                .ok_or_else(|| anyhow::anyhow!("non-tensor checkpoint value: {tensor_name}"))?;
            // This pinned Candle decoder stores a byte offset in TensorInfo's
            // Layout. Normalize it to element units before validating the view.
            let width = info.dtype.size_in_bytes();
            anyhow::ensure!(
                info.layout.start_offset().is_multiple_of(width),
                "misaligned tensor storage offset"
            );
            info.layout = candle_core::Layout::new(
                info.layout.shape().clone(),
                info.layout.stride().to_vec(),
                info.layout.start_offset() / width,
            );
            anyhow::ensure!(
                tensors.insert(tensor_name.into(), info).is_none(),
                "duplicate checkpoint tensor"
            );
        }
    }
    Ok(())
}
#[cfg(any(test, feature = "cosyvoice3", feature = "indextts"))]
fn validate_metadata(value: Object, depth: usize) -> anyhow::Result<()> {
    anyhow::ensure!(depth <= 8, "checkpoint metadata nesting exceeds budget");
    match value {
        Object::Int(_) | Object::Long(_) | Object::Unicode(_) | Object::Bool(_) | Object::None => {
            Ok(())
        }
        Object::Dict(items) => {
            for (key, value) in items {
                anyhow::ensure!(
                    matches!(key, Object::Unicode(_)),
                    "invalid checkpoint metadata key"
                );
                validate_metadata(value, depth + 1)?;
            }
            Ok(())
        }
        _ => anyhow::bail!("unsupported checkpoint metadata object"),
    }
}
#[cfg(any(test, feature = "cosyvoice3", feature = "indextts"))]
fn validate(info: &TensorInfo) -> anyhow::Result<()> {
    anyhow::ensure!(info.layout.dims().len() <= 8, "tensor rank exceeds budget");
    let elements = info
        .layout
        .dims()
        .iter()
        .try_fold(1usize, |count, &dim| count.checked_mul(dim))
        .ok_or_else(|| anyhow::anyhow!("tensor shape overflow"))?;
    let bytes = elements
        .checked_mul(info.dtype.size_in_bytes())
        .ok_or_else(|| anyhow::anyhow!("tensor size overflow"))?;
    anyhow::ensure!(
        bytes <= MAX_TENSOR_BYTES,
        "tensor exceeds conversion budget"
    );
    anyhow::ensure!(
        info.layout.is_contiguous() || info.layout.is_fortran_contiguous(),
        "unsupported tensor strides"
    );
    let end = info
        .layout
        .start_offset()
        .checked_add(elements)
        .ok_or_else(|| anyhow::anyhow!("tensor offset overflow"))?;
    anyhow::ensure!(end <= info.storage_size, "tensor exceeds declared storage");
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_preparation::Archive;
    const FIXTURE: &[u8] = include_bytes!("../../tests/fixtures/native-tensors.pt");
    const HASH: &str = "01b921749b21e8f10141d3f0d066b118e13bd47cb2ad9c8c681e6802fd290e7e";
    #[test]
    fn reads_nested_dtypes_views_and_element_offsets() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("synthetic.pt");
        std::fs::write(&path, FIXTURE).unwrap();
        let control = PreparationControl::default();
        let archive =
            Archive::open(&path, FIXTURE.len() as u64, HASH, &["model"], &control).unwrap();
        assert_eq!(archive.tensors().len(), 3);
        assert_eq!(
            archive
                .get("offset", &control)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [4., 5., 6., 7.]
        );
        assert_eq!(
            archive
                .get("nested.double", &control)
                .unwrap()
                .to_vec1::<f64>()
                .unwrap(),
            [0.5, 2.]
        );
        assert_eq!(
            archive
                .get("transpose", &control)
                .unwrap()
                .to_vec2::<f32>()
                .unwrap()[1],
            [1., 5., 9., 13.]
        );
    }
    #[test]
    fn rejects_unverified_input_before_decoding() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("untrusted.pt");
        std::fs::write(&path, b"arbitrary executable object").unwrap();
        assert!(Archive::open(&path, 27, HASH, &[], &PreparationControl::default()).is_err());
    }
    #[test]
    fn rejects_object_reduction_without_executing_it() {
        use sha2::{Digest, Sha256};
        use std::io::Write;
        let root = tempfile::tempdir().unwrap();
        let sentinel = root.path().join("must-not-exist");
        let expression = format!(
            "__import__('pathlib').Path({:?}).write_text('unsafe')",
            sentinel.to_str().unwrap()
        );
        let mut pickle = b"\x80\x02cbuiltins\neval\nX".to_vec();
        pickle.extend((expression.len() as u32).to_le_bytes());
        pickle.extend(expression.as_bytes());
        pickle.extend(b"\x85R.");
        let path = root.path().join("object.pt");
        let mut zip = zip::ZipWriter::new(std::fs::File::create_new(&path).unwrap());
        zip.start_file("archive/data.pkl", zip::write::SimpleFileOptions::default())
            .unwrap();
        zip.write_all(&pickle).unwrap();
        zip.finish().unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let sha = format!("{:x}", Sha256::digest(&bytes));
        assert!(
            Archive::open(
                &path,
                bytes.len() as u64,
                &sha,
                &[],
                &PreparationControl::default()
            )
            .is_err()
        );
        assert!(!sentinel.exists());
    }
    #[test]
    fn rejects_shape_stride_storage_and_tensor_budget_violations_before_allocation() {
        let info = |dims: Vec<usize>, strides: Vec<usize>, offset, storage_size| TensorInfo {
            name: "synthetic".into(),
            dtype: candle_core::DType::F32,
            layout: candle_core::Layout::new(dims.into(), strides, offset),
            path: "archive/data/0".into(),
            storage_size,
        };
        assert!(validate(&info(vec![usize::MAX, 2], vec![2, 1], 0, usize::MAX)).is_err());
        assert!(validate(&info(vec![2, 2], vec![3, 1], 0, 8)).is_err());
        assert!(validate(&info(vec![3], vec![1], 5, 7)).is_err());
        assert!(validate(&info(vec![3], vec![1], usize::MAX, usize::MAX)).is_err());
        let over = MAX_TENSOR_BYTES / 4 + 1;
        assert!(validate(&info(vec![over], vec![1], 0, over)).is_err());
        assert!(validate(&info(vec![60509, 1280], vec![1280, 1], 0, 60509 * 1280)).is_ok());
    }
    #[test]
    fn missing_selection_and_cancel_do_not_read_payloads() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("synthetic.pt");
        std::fs::write(&path, FIXTURE).unwrap();
        let control = PreparationControl::default();
        assert!(Archive::open(&path, FIXTURE.len() as u64, HASH, &["missing"], &control).is_err());
        let archive =
            Archive::open(&path, FIXTURE.len() as u64, HASH, &["model"], &control).unwrap();
        control.cancel();
        assert!(archive.get("offset", &control).is_err());
    }
}

pub(crate) struct TensorSpec {
    pub shape: Vec<usize>,
    pub dtype: candle_core::DType,
}
/// Write deterministic safetensors while retaining at most one converted tensor.
pub(crate) fn write(
    path: &Path,
    specs: &BTreeMap<String, TensorSpec>,
    control: &PreparationControl,
    mut load: impl FnMut(&str) -> anyhow::Result<Tensor>,
) -> anyhow::Result<()> {
    use candle_core::{CpuStorage, DType, Storage};
    use std::io::{BufWriter, Write};
    let mut header = BTreeMap::new();
    let mut offset = 0u64;
    anyhow::ensure!(
        !specs.is_empty() && specs.len() <= MAX_ENTRIES,
        "invalid output tensor count"
    );
    for (name, spec) in specs {
        anyhow::ensure!(
            name.len() <= 1024 && !name.is_empty() && spec.shape.len() <= 8,
            "invalid output tensor descriptor"
        );
        let count = spec
            .shape
            .iter()
            .try_fold(1usize, |count, &dim| count.checked_mul(dim))
            .ok_or_else(|| anyhow::anyhow!("output shape overflow"))?;
        let bytes = count
            .checked_mul(spec.dtype.size_in_bytes())
            .ok_or_else(|| anyhow::anyhow!("output size overflow"))?;
        anyhow::ensure!(bytes <= MAX_TENSOR_BYTES, "output tensor exceeds budget");
        let end = offset
            .checked_add(bytes as u64)
            .ok_or_else(|| anyhow::anyhow!("output offset overflow"))?;
        anyhow::ensure!(
            end <= 8 * 1024 * 1024 * 1024,
            "output component exceeds budget"
        );
        let dtype = match spec.dtype {
            DType::F32 => "F32",
            DType::F64 => "F64",
            _ => anyhow::bail!("unsupported output dtype"),
        };
        header.insert(
            name,
            serde_json::json!({"dtype":dtype,"shape":spec.shape,"data_offsets":[offset,end]}),
        );
        offset = end;
    }
    let mut header = serde_json::to_vec(&header)?;
    anyhow::ensure!(
        header.len() as u64 <= MAX_METADATA,
        "output header exceeds budget"
    );
    header.resize(header.len().div_ceil(8) * 8, b' ');
    control.check()?;
    let mut output = BufWriter::with_capacity(64 * 1024, File::create_new(path)?);
    output.write_all(&(header.len() as u64).to_le_bytes())?;
    output.write_all(&header)?;
    for (name, spec) in specs {
        control.check()?;
        let tensor = load(name)?.contiguous()?;
        anyhow::ensure!(
            tensor.dtype() == spec.dtype && tensor.dims() == spec.shape,
            "converted tensor contract mismatch: {name}"
        );
        let (storage, layout) = tensor.storage_and_layout();
        let start = layout.start_offset();
        let end = start + tensor.elem_count();
        match &*storage {
            Storage::Cpu(CpuStorage::F32(values)) => {
                for chunk in values[start..end].chunks(16 * 1024) {
                    control.check()?;
                    for value in chunk {
                        anyhow::ensure!(value.is_finite(), "non-finite converted tensor");
                        output.write_all(&value.to_le_bytes())?;
                    }
                }
            }
            Storage::Cpu(CpuStorage::F64(values)) => {
                for chunk in values[start..end].chunks(8 * 1024) {
                    control.check()?;
                    for value in chunk {
                        anyhow::ensure!(value.is_finite(), "non-finite converted tensor");
                        output.write_all(&value.to_le_bytes())?;
                    }
                }
            }
            _ => anyhow::bail!("conversion output must be F32/F64 on CPU"),
        }
    }
    output.flush()?;
    output.get_ref().sync_all()?;
    Ok(())
}
#[cfg(test)]
mod output_tests {
    use super::*;
    use crate::native_preparation::{TensorSpec, write_tensors as write};
    use candle_core::DType;
    #[test]
    fn streaming_output_roundtrips_float_precisions_and_rejects_overflow() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("output.safetensors");
        let specs = BTreeMap::from([
            (
                "single".into(),
                TensorSpec {
                    shape: vec![2],
                    dtype: DType::F32,
                },
            ),
            (
                "double".into(),
                TensorSpec {
                    shape: vec![2],
                    dtype: DType::F64,
                },
            ),
        ]);
        write(&path, &specs, &PreparationControl::default(), |name| {
            Ok(if name == "single" {
                Tensor::new(&[1.5f32, 2.5], &Device::Cpu)?
            } else {
                Tensor::new(&[0.1f64, 0.2], &Device::Cpu)?
            })
        })
        .unwrap();
        let loaded = candle_core::safetensors::load(&path, &Device::Cpu).unwrap();
        assert_eq!(loaded["double"].to_vec1::<f64>().unwrap(), [0.1, 0.2]);
        assert_eq!(loaded["single"].to_vec1::<f32>().unwrap(), [1.5, 2.5]);
        let invalid = BTreeMap::from([(
            "big".into(),
            TensorSpec {
                shape: vec![usize::MAX, 2],
                dtype: DType::F32,
            },
        )]);
        let invalid_path = root.path().join("bad.safetensors");
        assert!(
            write(
                &invalid_path,
                &invalid,
                &PreparationControl::default(),
                |_| panic!("must reject before loading")
            )
            .is_err()
        );
        assert!(!invalid_path.exists());
    }
}
#[cfg(test)]
mod real_inputs {
    use super::*;
    #[test]
    #[ignore = "explicit fixed official Kokoro checkpoint; no downloads"]
    fn pinned_kokoro_archive_reads_all_official_tensors() {
        let path = std::env::var_os("TALECHIME_KOKORO_CHECKPOINT")
            .expect("set TALECHIME_KOKORO_CHECKPOINT");
        let control = PreparationControl::default();
        let archive = Archive::open(
            &Path::new(&path).canonicalize().unwrap(),
            327247856,
            "b1d8410fa44dfb5c15471fd6c4225ea6b4e9ac7fa03c98e8bea47a9928476e2b",
            &[],
            &control,
        )
        .unwrap();
        assert_eq!(archive.tensors().len(), 548);
        for name in archive.tensors().keys() {
            archive.get(name, &control).unwrap();
        }
    }
}

/// Bounded file reader for pinned safetensors; does not mmap mutable user inputs.
#[cfg(any(test, feature = "kokoro"))]
pub(crate) struct SafeArchive {
    path: PathBuf,
    specs: BTreeMap<String, TensorSpec>,
    offsets: BTreeMap<String, u64>,
}
#[cfg(any(test, feature = "kokoro"))]
impl SafeArchive {
    pub fn open(
        path: &Path,
        size: u64,
        sha256: &str,
        control: &PreparationControl,
    ) -> anyhow::Result<Self> {
        #[derive(serde::Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Header {
            dtype: String,
            shape: Vec<usize>,
            data_offsets: [u64; 2],
        }
        anyhow::ensure!(
            plan::verified(path, size, sha256, control)?,
            "unverified safetensors input"
        );
        let mut file = File::open(path)?;
        let mut prefix = [0u8; 8];
        file.read_exact(&mut prefix)?;
        let length = u64::from_le_bytes(prefix);
        anyhow::ensure!(
            length <= MAX_METADATA && length.checked_add(8).is_some_and(|v| v <= size),
            "invalid safetensors header size"
        );
        let mut header = vec![0u8; length as usize];
        file.read_exact(&mut header)?;
        let entries: BTreeMap<String, serde_json::Value> = serde_json::from_slice(&header)?;
        anyhow::ensure!(
            !entries.is_empty() && entries.len() <= MAX_ENTRIES,
            "invalid safetensors entry count"
        );
        let mut specs = BTreeMap::new();
        let mut offsets = BTreeMap::new();
        let mut regions = Vec::new();
        for (name, value) in entries {
            if name == "__metadata__" {
                let _: BTreeMap<String, String> = serde_json::from_value(value)?;
                continue;
            }
            let entry: Header = serde_json::from_value(value)?;
            let dtype = match entry.dtype.as_str() {
                "F32" => candle_core::DType::F32,
                "F64" => candle_core::DType::F64,
                _ => anyhow::bail!("unsupported safetensors conversion dtype"),
            };
            anyhow::ensure!(
                name.len() <= 1024 && entry.shape.len() <= 8,
                "invalid safetensors tensor descriptor"
            );
            let count = entry
                .shape
                .iter()
                .try_fold(1usize, |n, &d| n.checked_mul(d))
                .ok_or_else(|| anyhow::anyhow!("safetensors shape overflow"))?;
            let bytes = count
                .checked_mul(dtype.size_in_bytes())
                .ok_or_else(|| anyhow::anyhow!("safetensors size overflow"))?;
            anyhow::ensure!(
                bytes <= MAX_TENSOR_BYTES
                    && entry.data_offsets[0].checked_add(bytes as u64)
                        == Some(entry.data_offsets[1]),
                "invalid safetensors tensor range"
            );
            let end = length
                .checked_add(8)
                .and_then(|n| n.checked_add(entry.data_offsets[1]))
                .ok_or_else(|| anyhow::anyhow!("safetensors offset overflow"))?;
            anyhow::ensure!(end <= size, "safetensors tensor exceeds file");
            regions.push(entry.data_offsets);
            offsets.insert(name.clone(), length + 8 + entry.data_offsets[0]);
            specs.insert(
                name,
                TensorSpec {
                    dtype,
                    shape: entry.shape,
                },
            );
        }
        regions.sort_unstable();
        let mut end = 0u64;
        for region in regions {
            anyhow::ensure!(
                region[0] == end,
                "safetensors contains overlapping or missing data"
            );
            end = region[1];
        }
        anyhow::ensure!(
            end + length + 8 == size && !specs.is_empty(),
            "safetensors data size mismatch"
        );
        Ok(Self {
            path: path.into(),
            specs,
            offsets,
        })
    }
    pub fn tensors(&self) -> &BTreeMap<String, TensorSpec> {
        &self.specs
    }
    pub fn get(&self, name: &str, control: &PreparationControl) -> anyhow::Result<Tensor> {
        use std::io::{Seek, SeekFrom};
        let spec = self
            .specs
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing safetensors tensor: {name}"))?;
        let bytes = spec.shape.iter().product::<usize>() * spec.dtype.size_in_bytes();
        let mut data = Vec::new();
        data.try_reserve_exact(bytes)?;
        data.resize(bytes, 0);
        let mut file = File::open(&self.path)?;
        file.seek(SeekFrom::Start(self.offsets[name]))?;
        for chunk in data.chunks_mut(64 * 1024) {
            control.check()?;
            file.read_exact(chunk)?;
        }
        Ok(Tensor::from_raw_buffer(
            &data,
            spec.dtype,
            &spec.shape,
            &Device::Cpu,
        )?)
    }
}
#[cfg(test)]
mod safe_tests {
    use super::*;
    use sha2::{Digest, Sha256};
    #[test]
    fn reads_independent_safetensors_and_rejects_bad_ranges_before_allocation() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("input.safetensors");
        let value = Tensor::new(&[1.25f32, 2.5], &Device::Cpu).unwrap();
        candle_core::safetensors::save(&std::collections::HashMap::from([("value", value)]), &path)
            .unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        let control = PreparationControl::default();
        let archive = SafeArchive::open(&path, bytes.len() as u64, &hash, &control).unwrap();
        assert_eq!(archive.tensors().len(), 1);
        assert_eq!(
            archive
                .get("value", &control)
                .unwrap()
                .to_vec1::<f32>()
                .unwrap(),
            [1.25, 2.5]
        );
        let header =
            br#"{"bad":{"dtype":"F32","shape":[18446744073709551615,2],"data_offsets":[0,8]}}"#;
        let mut bad = (header.len() as u64).to_le_bytes().to_vec();
        bad.extend_from_slice(header);
        bad.extend_from_slice(&[0; 8]);
        std::fs::write(&path, &bad).unwrap();
        let hash = format!("{:x}", Sha256::digest(&bad));
        assert!(SafeArchive::open(&path, bad.len() as u64, &hash, &control).is_err());
    }
}
