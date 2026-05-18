use core::fmt;

/// Errors returned by safe wrapper helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// A caller-provided buffer was too small for the requested image transfer.
    InvalidLength {
        /// Expected byte count for the requested image transfer.
        expected: usize,
        /// Actual byte count supplied by the caller.
        actual: usize,
    },
}

/// Convenient result alias used throughout the crate.
pub type Result<T> = core::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidLength { expected, actual } => {
                write!(
                    f,
                    "buffer too small: expected at least {expected} bytes, got {actual}"
                )
            }
        }
    }
}

impl std::error::Error for Error {}
