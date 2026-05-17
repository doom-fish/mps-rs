use crate::ffi;
use apple_metal::{CommandBuffer, MetalBuffer, MetalDevice};
use core::ffi::c_void;
use core::ptr;

/// Selected `MPSDataType` constants used for matrices and vectors.
pub mod data_type {
    pub const INVALID: u32 = 0;
    pub const FLOAT32: u32 = 0x1000_0020;
    pub const FLOAT16: u32 = 0x1000_0010;
    pub const INT8: u32 = 0x2000_0008;
    pub const INT16: u32 = 0x2000_0010;
    pub const INT32: u32 = 0x2000_0020;
    pub const UINT8: u32 = 0x0000_0008;
    pub const UINT16: u32 = 0x0000_0010;
    pub const UINT32: u32 = 0x0000_0020;
    pub const UNORM8: u32 = 0x4000_0008;
}

/// Return the byte width of a supported `MPSDataType`.
#[must_use]
pub const fn data_type_size(data_type: u32) -> Option<usize> {
    match data_type {
        data_type::FLOAT16 | data_type::INT16 | data_type::UINT16 => Some(2),
        data_type::FLOAT32 | data_type::INT32 | data_type::UINT32 => Some(4),
        data_type::INT8 | data_type::UINT8 | data_type::UNORM8 => Some(1),
        _ => None,
    }
}

/// Plain-Rust configuration for `MPSMatrixDescriptor`.
#[derive(Debug, Clone, Copy)]
pub struct MatrixDescriptor {
    pub rows: usize,
    pub columns: usize,
    pub matrices: usize,
    pub row_bytes: usize,
    pub matrix_bytes: usize,
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
}

/// Plain-Rust configuration for `MPSVectorDescriptor`.
#[derive(Debug, Clone, Copy)]
pub struct VectorDescriptor {
    pub length: usize,
    pub vectors: usize,
    pub vector_bytes: usize,
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
}

macro_rules! opaque_handle {
    ($name:ident) => {
        pub struct $name {
            ptr: *mut c_void,
        }

        // SAFETY: MPS handles are opaque pointers to thread-safe Swift/ObjC objects.
        unsafe impl Send for $name {}
        // SAFETY: MPS handles are opaque pointers to thread-safe Swift/ObjC objects.
        unsafe impl Sync for $name {}

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
            #[must_use]
            pub const fn as_ptr(&self) -> *mut c_void {
                self.ptr
            }
        }
    };
}

opaque_handle!(Matrix);
impl Matrix {
    /// Wrap an existing `MTLBuffer` as an `MPSMatrix`.
    #[must_use]
    pub fn new_with_buffer(buffer: &MetalBuffer, descriptor: MatrixDescriptor) -> Option<Self> {
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
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSMatrix` pointer while `self` is alive.
        unsafe { ffi::mps_matrix_rows(self.ptr) }
    }

    #[must_use]
    pub fn columns(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSMatrix` pointer while `self` is alive.
        unsafe { ffi::mps_matrix_columns(self.ptr) }
    }

    #[must_use]
    pub fn matrices(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSMatrix` pointer while `self` is alive.
        unsafe { ffi::mps_matrix_matrices(self.ptr) }
    }

    #[must_use]
    pub fn row_bytes(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSMatrix` pointer while `self` is alive.
        unsafe { ffi::mps_matrix_row_bytes(self.ptr) }
    }

    #[must_use]
    pub fn matrix_bytes(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSMatrix` pointer while `self` is alive.
        unsafe { ffi::mps_matrix_matrix_bytes(self.ptr) }
    }

    #[must_use]
    pub fn data_type(&self) -> u32 {
        // SAFETY: `self.ptr` is a valid `MPSMatrix` pointer while `self` is alive.
        unsafe { ffi::mps_matrix_data_type(self.ptr) }
    }
}

opaque_handle!(Vector);
pub use crate::generated::matrix::*;

impl Vector {
    /// Wrap an existing `MTLBuffer` as an `MPSVector`.
    #[must_use]
    pub fn new_with_buffer(buffer: &MetalBuffer, descriptor: VectorDescriptor) -> Option<Self> {
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
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn length(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSVector` pointer while `self` is alive.
        unsafe { ffi::mps_vector_length(self.ptr) }
    }

    #[must_use]
    pub fn vectors(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSVector` pointer while `self` is alive.
        unsafe { ffi::mps_vector_vectors(self.ptr) }
    }

    #[must_use]
    pub fn vector_bytes(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSVector` pointer while `self` is alive.
        unsafe { ffi::mps_vector_vector_bytes(self.ptr) }
    }

    #[must_use]
    pub fn data_type(&self) -> u32 {
        // SAFETY: `self.ptr` is a valid `MPSVector` pointer while `self` is alive.
        unsafe { ffi::mps_vector_data_type(self.ptr) }
    }
}

/// Plain-Rust configuration for `MPSMatrixMultiplication`.
#[derive(Debug, Clone, Copy)]
pub struct MatrixMultiplicationDescriptor {
    pub transpose_left: bool,
    pub transpose_right: bool,
    pub result_rows: usize,
    pub result_columns: usize,
    pub interior_columns: usize,
    pub alpha: f64,
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

opaque_handle!(MatrixMultiplication);
impl MatrixMultiplication {
    /// Build a configurable GEMM kernel with optional transposition and scaling.
    #[must_use]
    pub fn new(device: &MetalDevice, descriptor: MatrixMultiplicationDescriptor) -> Option<Self> {
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
            Some(Self { ptr })
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

    /// Encode the matrix multiplication onto a command buffer.
    pub fn encode(
        &self,
        command_buffer: &CommandBuffer,
        left: &Matrix,
        right: &Matrix,
        result: &Matrix,
    ) {
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
    }
}
