//! Candle-only MOSS adapters. Model identities and voices never migrate implicitly.
#[cfg(feature = "moss-nano-candle")]
mod audio;
pub mod candle;
#[cfg(feature = "moss-nano-candle")]
pub mod nano;
#[cfg(feature = "moss-nano-candle")]
mod prompt;
#[cfg(feature = "moss-nano-candle")]
mod text;
#[cfg(feature = "moss-nano-candle")]
pub mod voices;
#[cfg(feature = "moss-nano-candle")]
use talechime_protocol::Capabilities;
#[cfg(feature = "moss-nano-candle")]
fn nano_capabilities(directory: &std::path::Path) -> anyhow::Result<Capabilities> {
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("moss/assets/nano-prompt.json"))?;
    let mut voices = Vec::new();
    let mut names = std::collections::BTreeMap::new();
    for row in manifest["builtin_voices"].as_array().unwrap() {
        let id = row["voice"].as_str().unwrap().to_owned();
        names.insert(id.clone(), row["display_name"].as_str().unwrap().to_owned());
        voices.push(id);
    }
    for voice in voices::VoiceStore::new(directory).list()? {
        names.insert(voice.id.clone(), voice.name);
        voices.push(voice.id);
    }
    Ok(Capabilities {
        model: Some("nano-candle".into()),
        model_name: "MOSS Nano Candle".into(),
        backend: "moss".into(),
        voices,
        default_voice: "Weiguo".into(),
        voice_names: names,
        native_streaming: true,
        cloning: true,
        style: false,
        compiled_devices: Vec::new(),
        pronunciation: false,
        continuation: true,
        parameters: Vec::new(),
    })
}
