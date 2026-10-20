//! Device probes shared by the model adapters.
/// Probe before constructing Candle's device: 0.11 indexes an empty Metal
/// device list on GPU-less macOS hosts.
pub(crate) fn metal_is_available() -> bool {
    #[cfg(all(
        target_os = "macos",
        any(
            feature = "indextts-metal",
            feature = "cosyvoice3-metal",
            feature = "moss-nano-candle-metal",
            feature = "moss-candle-metal",
            feature = "qwen-metal",
            feature = "voxcpm-metal",
            feature = "omnivoice-metal",
        )
    ))]
    {
        !candle_metal_kernels::metal::Device::all().is_empty()
    }
    #[cfg(not(all(
        target_os = "macos",
        any(
            feature = "indextts-metal",
            feature = "cosyvoice3-metal",
            feature = "moss-nano-candle-metal",
            feature = "moss-candle-metal",
            feature = "qwen-metal",
            feature = "voxcpm-metal",
            feature = "omnivoice-metal",
        )
    )))]
    {
        false
    }
}
