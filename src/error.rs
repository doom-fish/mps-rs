use core::fmt;

/// Errors returned by safe wrapper helpers.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A caller-provided buffer was too small for the requested image transfer.
    InvalidLength {
        /// Expected byte count for the requested image transfer.
        expected: usize,
        /// Actual byte count supplied by the caller.
        actual: usize,
    },
    BufferTooSmall {
        required: usize,
        length: usize,
    },
    Misaligned {
        field: &'static str,
        value: usize,
        alignment: usize,
    },
    DimensionMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidArgument(&'static str),
    UnsupportedDataType(u32),
    Overflow,
    Unsupported(&'static str),
    Rejected(&'static str),
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
            Self::BufferTooSmall { required, length } => write!(
                f,
                "Metal buffer too small: {required} bytes required, buffer has {length}"
            ),
            Self::Misaligned {
                field,
                value,
                alignment,
            } => write!(f, "{field} {value} is not a multiple of {alignment}"),
            Self::DimensionMismatch {
                field,
                expected,
                actual,
            } => write!(f, "{field} mismatch: expected {expected}, got {actual}"),
            Self::InvalidArgument(message) => write!(f, "invalid argument: {message}"),
            Self::UnsupportedDataType(data_type) => {
                write!(f, "unsupported MPSDataType raw value {data_type:#x}")
            }
            Self::Overflow => f.write_str("size computation overflowed"),
            Self::Unsupported(message) => write!(f, "unsupported: {message}"),
            Self::Rejected(operation) => {
                write!(f, "Metal Performance Shaders rejected {operation}")
            }
        }
    }
}

impl std::error::Error for Error {}
