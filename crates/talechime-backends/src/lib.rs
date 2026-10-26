//! Concrete model adapters. The core and reader never import inference types.
#[cfg(feature = "asr")]
pub mod asr;
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice"
))]
mod cache_identity;
pub mod devices;
#[cfg(any(
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "asr"
))]
mod download;
#[cfg(feature = "indextts")]
pub mod indextts;
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
    feature = "cosyvoice3",
    feature = "omnivoice",
    feature = "asr",
    feature = "performance-tools",
))]
mod resources;
#[cfg(feature = "voxcpm")]
pub mod voxcpm;

#[cfg(feature = "cosyvoice3")]
pub mod cosyvoice3;

mod registry;
pub use registry::{PrepareOptions, Registry, ResolvedModel};
