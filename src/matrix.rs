use crate::error::{Error, Result};
use crate::ffi;
use apple_metal::{CommandBuffer, MetalBuffer, MetalDevice};
use core::ffi::c_void;
use core::ptr;

/// Selected `MPSDataType` constants used for matrices and vectors.
pub mod data_type {
    /// Wraps a `MPSDataType` raw value.
    pub const INVALID: u32 = 0;
    /// Wraps a `MPSDataType` raw value.
    pub const FLOAT32: u32 = 0x1000_0020;
    /// Wraps a `MPSDataType` raw value.
    pub const FLOAT16: u32 = 0x1000_0010;
    /// Wraps a `MPSDataType` raw value.
    pub const INT8: u32 = 0x2000_0008;
    /// Wraps a `MPSDataType` raw value.
    pub const INT16: u32 = 0x2000_0010;
    /// Wraps a `MPSDataType` raw value.
    pub const INT32: u32 = 0x2000_0020;
    /// Wraps a `MPSDataType` raw value.
    pub const UINT8: u32 = 0x0000_0008;
    /// Wraps a `MPSDataType` raw value.
    pub const UINT16: u32 = 0x0000_0010;
    /// Wraps a `MPSDataType` raw value.
    pub const UINT32: u32 = 0x0000_0020;
    /// Wraps a `MPSDataType` raw value.
    pub const UNORM8: u32 = 0x4000_0008;
    pub const BFLOAT16: u32 = 0x9000_0010;
    pub const COMPLEX_FLOAT32: u32 = 0x1100_0040;
    pub const COMPLEX_FLOAT16: u32 = 0x1100_0020;
    pub const COMPLEX_BFLOAT16: u32 = 0x9100_0020;
    pub const INT2: u32 = 0x2000_0002;
    pub const INT4: u32 = 0x2000_0004;
    pub const INT64: u32 = 0x2000_0040;
    pub const UINT2: u32 = 0x0000_0002;
    pub const UINT4: u32 = 0x0000_0004;
    pub const UINT64: u32 = 0x0000_0040;
    pub const BOOL: u32 = 0x8000_0008;
    pub const UNORM1: u32 = 0x4000_0001;
}

/// Return the byte width of a supported `MPSDataType`.
#[must_use]
pub const fn data_type_size(data_type: u32) -> Option<usize> {
    match data_type {
        data_type::FLOAT16 | data_type::INT16 | data_type::UINT16 | data_type::BFLOAT16 => Some(2),
        data_type::FLOAT32
        | data_type::INT32
        | data_type::UINT32
        | data_type::COMPLEX_FLOAT16
        | data_type::COMPLEX_BFLOAT16 => Some(4),
        data_type::INT8 | data_type::UINT8 | data_type::UNORM8 | data_type::BOOL => Some(1),
        data_type::INT64 | data_type::UINT64 | data_type::COMPLEX_FLOAT32 => Some(8),
        _ => None,
    }
}

/// Plain-Rust configuration for `MPSMatrixDescriptor`.
#[derive(Debug, Clone, Copy)]
pub struct MatrixDescriptor {
    /// Corresponds to the `rows` field on `MPSMatrixDescriptor`.
    pub rows: usize,
    /// Corresponds to the `columns` field on `MPSMatrixDescriptor`.
    pub columns: usize,
    /// Corresponds to the `matrices` field on `MPSMatrixDescriptor`.
    pub matrices: usize,
    /// Corresponds to the `row_bytes` field on `MPSMatrixDescriptor`.
    pub row_bytes: usize,
    /// Corresponds to the `matrix_bytes` field on `MPSMatrixDescriptor`.
    pub matrix_bytes: usize,
    /// Corresponds to the `data_type` field on `MPSMatrixDescriptor`.
    pub data_type: u32,
}

impl MatrixDescriptor {
    /// Construct a matrix descriptor with explicit row and matrix strides.
    #[must_use]
    pub const fn with_strides(
        rows: usize,
        columns: usize,
        matrices: usize,
        row_bytes: usize,
        matrix_bytes: usize,
        data_type: u32,
    ) -> Self {
        Self {
            rows,
            columns,
            matrices,
            row_bytes,
            matrix_bytes,
            data_type,
        }
    }

    /// Construct a single contiguous matrix descriptor for a supported data type.
    #[must_use]
    pub fn contiguous(rows: usize, columns: usize, data_type: u32) -> Option<Self> {
        let element_size = data_type_size(data_type)?;
        let row_bytes = columns.checked_mul(element_size)?;
        let matrix_bytes = rows.checked_mul(row_bytes)?;
        Some(Self::with_strides(
            rows,
            columns,
            1,
            row_bytes,
            matrix_bytes,
            data_type,
        ))
    }

    /// Query MPS's recommended row stride for a matrix width.
    #[must_use]
    pub fn recommended_row_bytes(columns: usize, data_type: u32) -> usize {
        // SAFETY: Pure function over scalar inputs.
        unsafe { ffi::mps_matrix_descriptor_row_bytes_for_columns(columns, data_type) }
    }

    pub fn required_buffer_length(&self) -> Result<usize> {
        let element_size =
            data_type_size(self.data_type).ok_or(Error::UnsupportedDataType(self.data_type))?;
        fit_native_int(&[
            self.rows,
            self.columns,
            self.matrices,
            self.row_bytes,
            self.matrix_bytes,
        ])?;
        if self.row_bytes % element_size != 0 {
            return Err(Error::Misaligned {
                field: "row_bytes",
                value: self.row_bytes,
                alignment: element_size,
            });
        }
        let row = self
            .columns
            .checked_mul(element_size)
            .ok_or(Error::Overflow)?;
        if self.row_bytes < row {
            return Err(Error::DimensionMismatch {
                field: "row_bytes",
                expected: row,
                actual: self.row_bytes,
            });
        }
        if self.matrices > 1 {
            let matrix = self
                .rows
                .checked_mul(self.row_bytes)
                .ok_or(Error::Overflow)?;
            if self.matrix_bytes < matrix {
                return Err(Error::DimensionMismatch {
                    field: "matrix_bytes",
                    expected: matrix,
                    actual: self.matrix_bytes,
                });
            }
            if self.row_bytes > 0 && self.matrix_bytes % self.row_bytes != 0 {
                return Err(Error::Misaligned {
                    field: "matrix_bytes",
                    value: self.matrix_bytes,
                    alignment: self.row_bytes,
                });
            }
        }
        if self.rows == 0 || self.columns == 0 || self.matrices == 0 {
            return Ok(0);
        }
        (self.matrices - 1)
            .checked_mul(self.matrix_bytes)
            .zip((self.rows - 1).checked_mul(self.row_bytes))
            .and_then(|(matrices, rows)| matrices.checked_add(rows))
            .and_then(|start| start.checked_add(row))
            .filter(|length| isize::try_from(*length).is_ok())
            .ok_or(Error::Overflow)
    }
}

fn fit_native_int(values: &[usize]) -> Result<()> {
    if values.iter().all(|value| isize::try_from(*value).is_ok()) {
        Ok(())
    } else {
        Err(Error::Overflow)
    }
}

fn ensure_buffer_holds(buffer: &MetalBuffer, required: usize) -> Result<()> {
    let length = buffer.length();
    if required > length {
        Err(Error::BufferTooSmall { required, length })
    } else {
        Ok(())
    }
}

/// Plain-Rust configuration for `MPSVectorDescriptor`.
#[derive(Debug, Clone, Copy)]
pub struct VectorDescriptor {
    /// Corresponds to the `length` field on `MPSVectorDescriptor`.
    pub length: usize,
    /// Corresponds to the `vectors` field on `MPSVectorDescriptor`.
    pub vectors: usize,
    /// Corresponds to the `vector_bytes` field on `MPSVectorDescriptor`.
    pub vector_bytes: usize,
    /// Corresponds to the `data_type` field on `MPSVectorDescriptor`.
    pub data_type: u32,
}

impl VectorDescriptor {
    /// Construct a vector descriptor with an explicit stride.
    #[must_use]
    pub const fn with_stride(
        length: usize,
        vectors: usize,
        vector_bytes: usize,
        data_type: u32,
    ) -> Self {
        Self {
            length,
            vectors,
            vector_bytes,
            data_type,
        }
    }

    /// Construct a contiguous vector descriptor for a supported data type.
    #[must_use]
    pub fn contiguous(length: usize, data_type: u32) -> Option<Self> {
        let element_size = data_type_size(data_type)?;
        let vector_bytes = length.checked_mul(element_size)?;
        Some(Self::with_stride(length, 1, vector_bytes, data_type))
    }

    /// Query MPS's recommended vector stride for a vector length.
    #[must_use]
    pub fn recommended_vector_bytes(length: usize, data_type: u32) -> usize {
        // SAFETY: Pure function over scalar inputs.
        unsafe { ffi::mps_vector_descriptor_vector_bytes_for_length(length, data_type) }
    }

    pub fn required_buffer_length(&self) -> Result<usize> {
        let element_size =
            data_type_size(self.data_type).ok_or(Error::UnsupportedDataType(self.data_type))?;
        fit_native_int(&[self.length, self.vectors, self.vector_bytes])?;
        let vector = self
            .length
            .checked_mul(element_size)
            .ok_or(Error::Overflow)?;
        if self.vectors > 1 {
            if self.vector_bytes % element_size != 0 {
                return Err(Error::Misaligned {
                    field: "vector_bytes",
                    value: self.vector_bytes,
                    alignment: element_size,
                });
            }
            if self.vector_bytes < vector {
                return Err(Error::DimensionMismatch {
                    field: "vector_bytes",
                    expected: vector,
                    actual: self.vector_bytes,
                });
            }
        }
        if self.length == 0 || self.vectors == 0 {
            return Ok(0);
        }
        (self.vectors - 1)
            .checked_mul(self.vector_bytes)
            .and_then(|start| start.checked_add(vector))
            .filter(|length| isize::try_from(*length).is_ok())
            .ok_or(Error::Overflow)
    }
}

macro_rules! opaque_handle {
    ($name:ident, $doc:expr, sync) => {
        opaque_handle!($name, $doc);

        // SAFETY: MPS handles are opaque pointers to thread-safe Swift/ObjC objects.
        unsafe impl Sync for $name {}
    };
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        pub struct $name {
            ptr: *mut c_void,
        }

        // SAFETY: MPS objects may move between threads; kernels and descriptors are used by one thread at a time.
        unsafe impl Send for $name {}

        impl Drop for $name {
            fn drop(&mut self) {
                if !self.ptr.is_null() {
                    // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
                    unsafe { ffi::mps_object_release(self.ptr) };
                    self.ptr = ptr::null_mut();
                }
            }
        }

        impl $name {
            /// Returns the retained Objective-C pointer backing this wrapper.
            #[must_use]
            pub const fn as_ptr(&self) -> *mut c_void {
                self.ptr
            }
        }
    };
}

opaque_handle!(Matrix, "Wraps `MPSMatrix`.", sync);
impl Matrix {
    /// Wrap an existing `MTLBuffer` as an `MPSMatrix`.
    pub fn new_with_buffer(buffer: &MetalBuffer, descriptor: MatrixDescriptor) -> Result<Self> {
        ensure_buffer_holds(buffer, descriptor.required_buffer_length()?)?;
        // SAFETY: `buffer` is a valid `MTLBuffer` wrapper and scalar parameters are POD.
        let ptr = unsafe {
            ffi::mps_matrix_new_with_buffer(
                buffer.as_ptr(),
                descriptor.rows,
                descriptor.columns,
                descriptor.matrices,
                descriptor.row_bytes,
                descriptor.matrix_bytes,
                descriptor.data_type,
            )
        };
        if ptr.is_null() {
            Err(Error::Rejected("MPSMatrix initWithBuffer"))
        } else {
            Ok(Self { ptr })
        }
    }

    /// Wraps the corresponding `MPSMatrix` method.
    #[must_use]
    pub fn rows(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSMatrix` pointer while `self` is alive.
        unsafe { ffi::mps_matrix_rows(self.ptr) }
    }

    /// Wraps the corresponding `MPSMatrix` method.
    #[must_use]
    pub fn columns(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSMatrix` pointer while `self` is alive.
        unsafe { ffi::mps_matrix_columns(self.ptr) }
    }

    /// Wraps the corresponding `MPSMatrix` method.
    #[must_use]
    pub fn matrices(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSMatrix` pointer while `self` is alive.
        unsafe { ffi::mps_matrix_matrices(self.ptr) }
    }

    /// Wraps the corresponding `MPSMatrix` method.
    #[must_use]
    pub fn row_bytes(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSMatrix` pointer while `self` is alive.
        unsafe { ffi::mps_matrix_row_bytes(self.ptr) }
    }

    /// Wraps the corresponding `MPSMatrix` method.
    #[must_use]
    pub fn matrix_bytes(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSMatrix` pointer while `self` is alive.
        unsafe { ffi::mps_matrix_matrix_bytes(self.ptr) }
    }

    /// Wraps the corresponding `MPSMatrix` method.
    #[must_use]
    pub fn data_type(&self) -> u32 {
        // SAFETY: `self.ptr` is a valid `MPSMatrix` pointer while `self` is alive.
        unsafe { ffi::mps_matrix_data_type(self.ptr) }
    }
}

opaque_handle!(Vector, "Wraps `MPSVector`.", sync);
#[doc(hidden)]
pub use crate::generated::matrix::*;

impl Vector {
    /// Wrap an existing `MTLBuffer` as an `MPSVector`.
    pub fn new_with_buffer(buffer: &MetalBuffer, descriptor: VectorDescriptor) -> Result<Self> {
        ensure_buffer_holds(buffer, descriptor.required_buffer_length()?)?;
        // SAFETY: `buffer` is a valid `MTLBuffer` wrapper and scalar parameters are POD.
        let ptr = unsafe {
            ffi::mps_vector_new_with_buffer(
                buffer.as_ptr(),
                descriptor.length,
                descriptor.vectors,
                descriptor.vector_bytes,
                descriptor.data_type,
            )
        };
        if ptr.is_null() {
            Err(Error::Rejected("MPSVector initWithBuffer"))
        } else {
            Ok(Self { ptr })
        }
    }

    /// Wraps the corresponding `MPSVector` method.
    #[must_use]
    pub fn length(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSVector` pointer while `self` is alive.
        unsafe { ffi::mps_vector_length(self.ptr) }
    }

    /// Wraps the corresponding `MPSVector` method.
    #[must_use]
    pub fn vectors(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSVector` pointer while `self` is alive.
        unsafe { ffi::mps_vector_vectors(self.ptr) }
    }

    /// Wraps the corresponding `MPSVector` method.
    #[must_use]
    pub fn vector_bytes(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSVector` pointer while `self` is alive.
        unsafe { ffi::mps_vector_vector_bytes(self.ptr) }
    }

    /// Wraps the corresponding `MPSVector` method.
    #[must_use]
    pub fn data_type(&self) -> u32 {
        // SAFETY: `self.ptr` is a valid `MPSVector` pointer while `self` is alive.
        unsafe { ffi::mps_vector_data_type(self.ptr) }
    }
}

/// Plain-Rust configuration for `MPSMatrixMultiplication`.
#[derive(Debug, Clone, Copy)]
pub struct MatrixMultiplicationDescriptor {
    /// Corresponds to the `transpose_left` field on `MPSMatrixMultiplication`.
    pub transpose_left: bool,
    /// Corresponds to the `transpose_right` field on `MPSMatrixMultiplication`.
    pub transpose_right: bool,
    /// Corresponds to the `result_rows` field on `MPSMatrixMultiplication`.
    pub result_rows: usize,
    /// Corresponds to the `result_columns` field on `MPSMatrixMultiplication`.
    pub result_columns: usize,
    /// Corresponds to the `interior_columns` field on `MPSMatrixMultiplication`.
    pub interior_columns: usize,
    /// Corresponds to the `alpha` field on `MPSMatrixMultiplication`.
    pub alpha: f64,
    /// Corresponds to the `beta` field on `MPSMatrixMultiplication`.
    pub beta: f64,
}

impl MatrixMultiplicationDescriptor {
    /// Construct the common `C = A * B` descriptor.
    #[must_use]
    pub const fn new(result_rows: usize, result_columns: usize, interior_columns: usize) -> Self {
        Self {
            transpose_left: false,
            transpose_right: false,
            result_rows,
            result_columns,
            interior_columns,
            alpha: 1.0,
            beta: 0.0,
        }
    }

    /// Construct a fully configurable descriptor.
    #[must_use]
    pub const fn with_options(
        transpose_left: bool,
        transpose_right: bool,
        result_rows: usize,
        result_columns: usize,
        interior_columns: usize,
        alpha: f64,
        beta: f64,
    ) -> Self {
        Self {
            transpose_left,
            transpose_right,
            result_rows,
            result_columns,
            interior_columns,
            alpha,
            beta,
        }
    }
}

/// Wraps `MPSMatrixMultiplication`.
pub struct MatrixMultiplication {
    ptr: *mut c_void,
    descriptor: MatrixMultiplicationDescriptor,
}

unsafe impl Send for MatrixMultiplication {}

impl Drop for MatrixMultiplication {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
            unsafe { ffi::mps_object_release(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl MatrixMultiplication {
    /// Build a configurable GEMM kernel with optional transposition and scaling.
    #[must_use]
    pub fn new(device: &MetalDevice, descriptor: MatrixMultiplicationDescriptor) -> Option<Self> {
        fit_native_int(&[
            descriptor.result_rows,
            descriptor.result_columns,
            descriptor.interior_columns,
        ])
        .ok()?;
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe {
            ffi::mps_matrix_multiplication_new(
                device.as_ptr(),
                descriptor.transpose_left,
                descriptor.transpose_right,
                descriptor.result_rows,
                descriptor.result_columns,
                descriptor.interior_columns,
                descriptor.alpha,
                descriptor.beta,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr, descriptor })
        }
    }

    /// Convenience constructor for the common `C = A * B` case.
    #[must_use]
    pub fn new_simple(
        device: &MetalDevice,
        result_rows: usize,
        result_columns: usize,
        interior_columns: usize,
    ) -> Option<Self> {
        Self::new(
            device,
            MatrixMultiplicationDescriptor::new(result_rows, result_columns, interior_columns),
        )
    }

    /// Returns the retained Objective-C pointer backing this wrapper.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    #[must_use]
    pub const fn descriptor(&self) -> MatrixMultiplicationDescriptor {
        self.descriptor
    }

    /// Encode the matrix multiplication onto a command buffer.
    pub fn encode(
        &self,
        command_buffer: &CommandBuffer,
        left: &Matrix,
        right: &Matrix,
        result: &Matrix,
    ) -> Result<()> {
        let descriptor = self.descriptor;
        let (m, n, k) = (
            descriptor.result_rows,
            descriptor.result_columns,
            descriptor.interior_columns,
        );
        let left_shape = if descriptor.transpose_left {
            (k, m)
        } else {
            (m, k)
        };
        let right_shape = if descriptor.transpose_right {
            (n, k)
        } else {
            (k, n)
        };
        ensure_covers(left, ("left rows", "left columns"), left_shape)?;
        ensure_covers(right, ("right rows", "right columns"), right_shape)?;
        ensure_covers(result, ("result rows", "result columns"), (m, n))?;
        if !multiplication_types_supported(left.data_type(), right.data_type(), result.data_type())
        {
            return Err(Error::InvalidArgument(
                "MPSMatrixMultiplication does not support this combination of data types",
            ));
        }
        // SAFETY: All handles come from safe wrappers and remain alive for the call.
        unsafe {
            ffi::mps_matrix_multiplication_encode(
                self.ptr,
                command_buffer.as_ptr(),
                left.as_ptr(),
                right.as_ptr(),
                result.as_ptr(),
            );
        };
        Ok(())
    }
}

fn ensure_covers(
    matrix: &Matrix,
    fields: (&'static str, &'static str),
    (rows, columns): (usize, usize),
) -> Result<()> {
    if matrix.rows() < rows {
        return Err(Error::DimensionMismatch {
            field: fields.0,
            expected: rows,
            actual: matrix.rows(),
        });
    }
    if matrix.columns() < columns {
        return Err(Error::DimensionMismatch {
            field: fields.1,
            expected: columns,
            actual: matrix.columns(),
        });
    }
    Ok(())
}

const fn multiplication_types_supported(left: u32, right: u32, result: u32) -> bool {
    use data_type::{FLOAT16, FLOAT32, INT16, INT8};
    matches!(
        (left, right, result),
        (FLOAT32, FLOAT32 | FLOAT16, FLOAT32)
            | (FLOAT16, FLOAT16, FLOAT16 | FLOAT32)
            | (INT8, INT8, FLOAT16 | FLOAT32)
            | (INT16, INT16, FLOAT32)
    )
}
