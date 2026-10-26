//! Provider selection is explicit; registration never implies measured acceleration.
pub mod calibration;
#[cfg(any(
    feature = "moss-nano-candle-metal",
    feature = "moss",
    feature = "qwen",
    feature = "voxcpm",
    feature = "omnivoice",
    feature = "cosyvoice3",
    feature = "indextts",
))]
pub(crate) mod metal;
