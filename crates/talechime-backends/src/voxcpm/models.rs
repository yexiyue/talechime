//! Immutable model identities; experimental weights never replace existing defaults.
use candle_core::{DType, Device};
use std::path::{Path, PathBuf};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Model {
    Q8,
    OriginalBf16,
}

pub(crate) enum ModelFiles {
    Q8 {
        base: PathBuf,
        acoustic: PathBuf,
    },
    Original {
        config: PathBuf,
        tokenizer: PathBuf,
        model: PathBuf,
        vae: PathBuf,
    },
}
impl ModelFiles {
    pub(crate) fn from_resolved(
        model: Model,
        mut paths: std::collections::BTreeMap<String, PathBuf>,
    ) -> anyhow::Result<Self> {
        let mut take = |name: &str| {
            paths
                .remove(name)
                .ok_or_else(|| anyhow::anyhow!("missing VoxCPM resource {name}"))
        };
        Ok(match model {
            Model::Q8 => Self::Q8 {
                base: take("VoxCPM2-BaseLM-Q8_0.gguf")?,
                acoustic: take("VoxCPM2-Acoustic-F16.gguf")?,
            },
            Model::OriginalBf16 => Self::Original {
                config: take("config.json")?,
                tokenizer: take("tokenizer.json")?,
                model: take("model.safetensors")?,
                vae: take("audiovae.pth")?,
            },
        })
    }
}
impl Model {
    pub fn parse(id: Option<&str>) -> anyhow::Result<Self> {
        match id {
            None | Some("2b-q8_0") => Ok(Self::Q8),
            Some("2b-bf16") => Ok(Self::OriginalBf16),
            _ => anyhow::bail!("unknown VoxCPM2 model {id:?}"),
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Self::Q8 => "2b-q8_0",
            Self::OriginalBf16 => "2b-bf16",
        }
    }
    pub fn revision(self) -> &'static str {
        match self {
            Self::Q8 => super::resources::REVISION,
            Self::OriginalBf16 => "32279effe8c19989596f05d353d1447f51d9e915",
        }
    }
    pub fn calibration_revision(self) -> &'static str {
        match self {
            Self::Q8 => super::CALIBRATION_REVISION,
            Self::OriginalBf16 => {
                "32279effe8c19989596f05d353d1447f51d9e915-candle-original-bf16-v1"
            }
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Q8 => "VoxCPM2 2B · Q8_0",
            Self::OriginalBf16 => "VoxCPM2 2B · 原始 BF16（实验）",
        }
    }
    pub fn directory(self, root: &Path) -> PathBuf {
        root.join("voxcpm/models")
            .join(self.id())
            .join(self.revision())
    }
    pub fn compiled_devices(self) -> Vec<talechime_protocol::Device> {
        super::compiled_devices()
            .into_iter()
            .filter(|device| self == Self::Q8 || *device == talechime_protocol::Device::Cuda)
            .collect()
    }
    pub(crate) fn manifest(self) -> &'static str {
        match self {
            Self::Q8 => include_str!("resources.json"),
            Self::OriginalBf16 => include_str!("original-resources.json"),
        }
    }
    pub(super) fn dtype(self, device: &Device) -> DType {
        match self {
            Self::OriginalBf16 => DType::BF16,
            Self::Q8 if device.is_cpu() => DType::F32,
            Self::Q8 => DType::F16,
        }
    }
    pub(super) fn load_files(
        self,
        files: &ModelFiles,
        device: &Device,
        cancel: &impl Fn() -> bool,
    ) -> candle_core::Result<talechime_voxcpm::Model> {
        match (self, files) {
            (Self::Q8, ModelFiles::Q8 { base, acoustic }) => {
                talechime_voxcpm::Model::load_files_with_cancel(base, acoustic, device, cancel)
            }
            (
                Self::OriginalBf16,
                ModelFiles::Original {
                    config,
                    tokenizer,
                    model,
                    vae,
                },
            ) => {
                if !device.is_cuda() {
                    candle_core::bail!("experimental VoxCPM2 BF16 requires CUDA");
                }
                talechime_voxcpm::Model::load_original_files(
                    config,
                    tokenizer,
                    model,
                    vae,
                    device,
                    DType::BF16,
                    cancel,
                )
            }
            _ => candle_core::bail!("VoxCPM model and resource files do not match"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn legacy_default_and_experimental_identity_are_separate() {
        assert_eq!(Model::parse(None).unwrap(), Model::Q8);
        assert!(Model::parse(Some("unknown")).is_err());
        assert_ne!(
            Model::Q8.directory(Path::new("cache")),
            Model::OriginalBf16.directory(Path::new("cache"))
        );
        assert!(
            !Model::OriginalBf16
                .compiled_devices()
                .contains(&talechime_protocol::Device::Cpu)
        );
        assert!(
            Model::OriginalBf16
                .load_files(
                    &ModelFiles::Original {
                        config: "missing".into(),
                        tokenizer: "missing".into(),
                        model: "missing".into(),
                        vae: "missing".into()
                    },
                    &Device::Cpu,
                    &|| false
                )
                .err()
                .unwrap()
                .to_string()
                .contains("requires CUDA")
        );
        let resources: Vec<crate::resources::Resource> =
            serde_json::from_str(Model::OriginalBf16.manifest()).unwrap();
        assert_eq!(resources.len(), 4);
        assert!(
            resources
                .iter()
                .all(|file| file.revision == Model::OriginalBf16.revision()
                    && file.repo.contains('/')
                    && file.size > 0)
        );
    }
}
