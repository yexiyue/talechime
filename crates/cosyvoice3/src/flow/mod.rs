//! Flow decoder: pre-lookahead upsampling + DiT conditional flow matching.

pub mod dit;
pub mod embeddings;
pub mod flow_matching;
#[cfg(all(feature = "metal", target_os = "macos"))]
pub(crate) mod metal_fused;
pub mod pre_lookahead;

pub use dit::DiT;
pub use flow_matching::{CausalConditionalCfm, CausalMaskedDiffWithDiT};
