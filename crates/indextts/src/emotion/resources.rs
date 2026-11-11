//! Fixed classifier identity; direct model-crate use also verifies these files.
use candle_core::Result;
use sha2::{Digest, Sha256};
use std::{fs::File, io::Read, path::Path};
pub(super) const FILES: &[(&str, u64, &str)] = &[
    (
        "qwen0.6bemo4-merge/chat_template.jinja",
        550,
        "a13337ea164ac47b2c1e644b6338d5e7e91cdcb58218511899c083112472730c",
    ),
    (
        "qwen0.6bemo4-merge/config.json",
        727,
        "2e04bc223dfb548dde52faebe2a819964523ff749cbe26467c49380aa371594f",
    ),
    (
        "qwen0.6bemo4-merge/generation_config.json",
        117,
        "6453260cdb41552215d87400ab399a760f25509ec2860ceb07817b195651626d",
    ),
    (
        "qwen0.6bemo4-merge/model.safetensors",
        1192135096,
        "11293257a8df593c154a8ecd5fc039f3076de35411e35f06d41b471e136f6641",
    ),
    (
        "qwen0.6bemo4-merge/tokenizer.json",
        11422654,
        "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    ),
    (
        "qwen0.6bemo4-merge/tokenizer_config.json",
        5433,
        "147bae7844ddfb8767c1f9cc0f505d4a26839bd6fbb0656d6d6bc17c82e9108a",
    ),
];
pub(super) fn verify(root: &Path) -> Result<()> {
    for &(name, size, expected) in FILES {
        let path = root.join(name);
        let mut file = File::open(&path).map_err(|error| {
            candle_core::Error::Msg(format!(
                "QwenEmotion requires fixed resource {}: {error}",
                path.display()
            ))
        })?;
        if file.metadata()?.len() != size {
            candle_core::bail!("QwenEmotion resource size mismatch: {name}");
        }
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
            candle_core::bail!("QwenEmotion resource identity mismatch: {name}");
        }
    }
    Ok(())
}
