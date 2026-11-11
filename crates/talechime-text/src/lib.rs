//! Low-level FST execution and token serialization; model frontends own policy.
mod fst;
mod reorder;
pub mod wetext;
pub use fst::FstNormalizer;
pub use reorder::{reorder, reorder_linux};

/// Text resource, graph or tagged-value errors.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The owning request stopped text processing.
    #[error("text processing cancelled")]
    Cancelled,
    /// Invalid grammar or tagged text.
    #[error("{0}")]
    Invalid(String),
    /// Resource could not be read.
    #[error(transparent)]
    Io(#[from] std::io::Error),
}
/// Result of a native text stage.
pub type Result<T> = std::result::Result<T, Error>;
macro_rules! fail {
    ($($arg:tt)*) => { return Err(crate::Error::Invalid(format!($($arg)*))) };
}
pub(crate) use fail;
