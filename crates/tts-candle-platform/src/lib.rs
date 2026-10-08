//! Target-scoped Cargo feature activation for one workspace Candle version.
//!
//! Candle's GPU features are not target-gated upstream. Routing them through
//! optional platform dependencies prevents `--all-features` on Windows from
//! building Objective-C, and prevents macOS builds from invoking nvcc.
//! Transformers' CUDA feature activates both core and NN; NN's Metal feature
//! activates core and Metal kernels. This crate contains no inference code.

/// Probe before constructing Candle's device: 0.11 indexes the Metal device list
/// without checking whether the host exposes a GPU (hosted macOS VMs may not).
pub fn metal_is_available() -> bool {
    #[cfg(all(feature = "metal", target_os = "macos"))]
    {
        !candle_metal_kernels::metal::Device::all().is_empty()
    }
    #[cfg(not(all(feature = "metal", target_os = "macos")))]
    {
        false
    }
}
