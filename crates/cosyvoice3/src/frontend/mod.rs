//! Frontend feature extraction and (optional) official ONNX execution.

pub mod audio;
pub mod text;

#[cfg(feature = "ort")]
pub mod ort_models;
