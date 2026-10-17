//! Flow decoder: pre-lookahead upsampling + DiT conditional flow matching.

pub mod dit;
pub mod embeddings;
pub mod flow_matching;
pub mod pre_lookahead;

pub use dit::DiT;
pub use flow_matching::{CausalConditionalCfm, CausalMaskedDiffWithDiT};
