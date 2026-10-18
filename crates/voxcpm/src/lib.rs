//! VoxCPM2 using the workspace Candle runtime.
//!
//! The production adapter uses the existing GGUF path. Original Safetensors
//! weights are available for explicit precision and performance qualification.
mod acoustic;
mod attention;
mod codec;
#[cfg(all(feature = "cuda", any(target_os = "windows", target_os = "linux")))]
mod cuda_graph;
mod model;
mod timing;
mod transformer;
#[cfg(test)]
mod validation;
mod weights;
#[cfg(feature = "experimental-dual-reference")]
pub use model::DualReference;
pub use model::{CudaGraphStatus, Model, Options, Outcome, Reference, SAMPLE_RATE};
pub use timing::Timings;
pub mod tokenizer;
