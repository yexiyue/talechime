//! Fixed-buffer CUDA graph capture and replay for Candle tensor programs.
//!
//! The recorder is independent of the model being recorded: a caller supplies
//! persistent input buffers, a warm-up result and a closure that runs the
//! operation, and receives a [`Replay`] that re-executes the recorded kernels
//! from the same device addresses. Model policy — what to capture, when to
//! capture, cache keys, fallback and cancellation — stays in the model crate.
//!
//! Consumers must own every tensor whose address the graph touches, destroy the
//! graph before releasing those tensors, and consume a replay output before the
//! next launch overwrites it.
#![cfg(any(target_os = "windows", target_os = "linux"))]

mod replay;

pub use cudarc;
pub use replay::{Replay, copy_into, driver_error, is_capturing};
