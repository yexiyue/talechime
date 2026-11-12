//! Concrete model adapters. The core and reader never import inference types.
/// Shared text/conditioning semantics. Bump when a change can alter saved speech
/// without changing the selected model's immutable resource revision.
pub const TEXT_SEMANTICS_REVISION: &str = "official-backend-text-v3";
#[cfg(feature = "asr")]
pub mod asr;
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice"
))]
mod cache_identity;
pub mod design;
pub mod devices;
pub mod downloads;
#[cfg(feature = "indextts")]
pub mod indextts;
#[cfg(feature = "kokoro")]
pub mod kokoro;
#[cfg(feature = "moss")]
pub mod moss;
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "cosyvoice3",
    feature = "indextts"
))]
mod observation;
#[cfg(feature = "omnivoice")]
pub mod omnivoice;
#[cfg(feature = "performance-tools")]
pub mod performance;
#[cfg(feature = "qwen")]
pub mod qwen;
#[cfg(any(
    feature = "qwen",
    feature = "voxcpm",
    feature = "cosyvoice3",
    feature = "indextts",
    feature = "omnivoice",
    feature = "moss"
))]
pub mod reference;
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "asr",
    feature = "performance-tools",
    feature = "kokoro",
    feature = "cosyvoice3",
    feature = "indextts",
))]
mod resources;
#[cfg(feature = "voxcpm")]
pub mod voxcpm;

#[cfg(feature = "cosyvoice3")]
pub mod cosyvoice3;

mod registry;
pub use registry::{PrepareOptions, ReferenceVoice, Registry, ResolvedModel, ResourceInventory};
mod resource_state;
mod resource_worker;
pub use resource_state::ResourceStatus;
