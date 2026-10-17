//! Frontend feature extraction and (optional) official ONNX execution.

pub mod audio;

#[cfg(feature = "ort")]
pub mod ort_models;
