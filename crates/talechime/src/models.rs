//! Human and JSON model discovery over the public facade catalog.
use std::path::Path;

pub fn run(
    root: &Path,
    backend: Option<&str>,
    model: Option<&str>,
    json: bool,
) -> anyhow::Result<()> {
    let catalog = talechime::model_catalog(root)?;
    let models: Vec<_> = catalog
        .into_iter()
        .filter(|info| {
            backend.is_none_or(|backend| info.capabilities.backend == backend)
                && model.is_none_or(|model| info.capabilities.model.as_deref() == Some(model))
        })
        .collect();
    if models.is_empty() && (backend.is_some() || model.is_some()) {
        anyhow::bail!(
            "no matching model in this build; run `talechime models` to list available selections"
        );
    }
    if json {
        println!("{}", serde_json::to_string_pretty(&models)?);
    } else if models.is_empty() {
        println!("no synthesis model is compiled into this build");
    } else {
        for info in models {
            let caps = info.capabilities;
            println!(
                "{}/{} — {}",
                caps.backend,
                caps.model.as_deref().unwrap_or("default"),
                caps.model_name
            );
            let devices = info
                .available_devices
                .iter()
                .map(|device| format!("{device:?}").to_lowercase())
                .collect::<Vec<_>>()
                .join(", ");
            println!(
                "  devices: {}",
                if devices.is_empty() {
                    "none available"
                } else {
                    &devices
                }
            );
            println!(
                "  voices: {}",
                if caps.voices.is_empty() {
                    "import or design a library voice".into()
                } else {
                    caps.voices.join(", ")
                }
            );
            println!(
                "  reference cloning: {}; style: {}",
                caps.cloning, caps.style
            );
            if !caps.parameters.is_empty() {
                println!(
                    "  parameters: {} (use --json for limits and defaults)",
                    caps.parameters
                        .iter()
                        .map(|param| param.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }
        }
    }
    Ok(())
}
