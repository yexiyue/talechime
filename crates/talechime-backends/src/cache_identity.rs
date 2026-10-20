//! Identities for immutable model resources and their concrete execution path.
use std::path::Path;
use talechime_protocol::Device;

pub(crate) fn identity(
    implementation: &str,
    revision: &str,
    device: Device,
    directory: &Path,
) -> String {
    let directory = directory
        .canonicalize()
        .unwrap_or_else(|_| directory.to_owned());
    serde_json::json!({
        "implementation": implementation,
        "version": env!("CARGO_PKG_VERSION"),
        "revision": revision,
        "device": device,
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "directory": directory,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_paths_and_resource_revisions_have_distinct_identities() {
        let directory = Path::new("models");
        let base = identity("nano-candle", "revision-1", Device::Cpu, directory);
        assert_eq!(
            base,
            identity("nano-candle", "revision-1", Device::Cpu, directory)
        );
        for other in [
            identity("nano-onnx", "revision-1", Device::Cpu, directory),
            identity("nano-candle", "revision-2", Device::Cpu, directory),
            identity("nano-candle", "revision-1", Device::Metal, directory),
            identity("nano-candle", "revision-1", Device::Cpu, Path::new("other")),
        ] {
            assert_ne!(base, other);
        }
    }
}
