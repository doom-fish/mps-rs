use crate::ffi;
use apple_metal::{CommandBuffer as MetalCommandBuffer, MetalBuffer, MetalDevice};
use core::ffi::c_void;
use core::ptr;

macro_rules! opaque_handle {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
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
                    // SAFETY: `ptr` is a +1 retained MPS object owned by this wrapper.
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

#[doc(hidden)]
pub use crate::generated::ndarray::*;

opaque_handle!(NDArrayDescriptor, "Wraps `MPSNDArrayDescriptor`.");
impl NDArrayDescriptor {
    /// Wraps the corresponding `MPSNDArrayDescriptor` method.
    #[must_use]
    pub fn with_dimension_sizes(data_type: u32, dimension_sizes: &[usize]) -> Option<Self> {
        // SAFETY: dimension_sizes.as_ptr() is valid for dimension_sizes.len() elements.
        let ptr = unsafe {
            ffi::mps_ndarray_descriptor_new_with_dimension_sizes(
                data_type,
                dimension_sizes.len(),
                dimension_sizes.as_ptr(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps the corresponding `MPSNDArrayDescriptor` method.
    #[must_use]
    pub fn data_type(&self) -> u32 {
        // SAFETY: self.ptr is a valid NDArrayDescriptor.
        unsafe { ffi::mps_ndarray_descriptor_data_type(self.ptr) }
    }

    /// Wraps the corresponding `MPSNDArrayDescriptor` setter.
    pub fn set_data_type(&self, data_type: u32) {
        // SAFETY: self.ptr is a valid NDArrayDescriptor.
        unsafe { ffi::mps_ndarray_descriptor_set_data_type(self.ptr, data_type) };
    }

    /// Wraps the corresponding `MPSNDArrayDescriptor` method.
    #[must_use]
    pub fn number_of_dimensions(&self) -> usize {
        // SAFETY: self.ptr is a valid NDArrayDescriptor.
        unsafe { ffi::mps_ndarray_descriptor_number_of_dimensions(self.ptr) }
    }

    /// Wraps the corresponding `MPSNDArrayDescriptor` setter.
    pub fn set_number_of_dimensions(&self, number_of_dimensions: usize) {
        // SAFETY: self.ptr is a valid NDArrayDescriptor.
        unsafe {
            ffi::mps_ndarray_descriptor_set_number_of_dimensions(self.ptr, number_of_dimensions);
        };
    }

    /// Wraps the corresponding `MPSNDArrayDescriptor` method.
    #[must_use]
    pub fn length_of_dimension(&self, dimension_index: usize) -> usize {
        // SAFETY: self.ptr is a valid NDArrayDescriptor and dimension_index is in bounds.
        unsafe { ffi::mps_ndarray_descriptor_length_of_dimension(self.ptr, dimension_index) }
    }

    /// Wraps the corresponding `MPSNDArrayDescriptor` method.
    pub fn reshape_with_dimension_sizes(&self, dimension_sizes: &[usize]) {
        // SAFETY: dimension_sizes.as_ptr() is valid for dimension_sizes.len() elements.
        unsafe {
            ffi::mps_ndarray_descriptor_reshape_with_dimension_sizes(
                self.ptr,
                dimension_sizes.len(),
                dimension_sizes.as_ptr(),
            );
        };
    }

    /// Wraps the corresponding `MPSNDArrayDescriptor` method.
    pub fn transpose_dimension(&self, dimension_index: usize, other_dimension_index: usize) {
        // SAFETY: Both dimension indices are validated by MPS.
        unsafe {
            ffi::mps_ndarray_descriptor_transpose_dimension(
                self.ptr,
                dimension_index,
                other_dimension_index,
            );
        };
    }
}

opaque_handle!(NDArray, "Wraps `MPSNDArray`.");
impl NDArray {
    /// Wraps a constructor on `MPSNDArray`.
    #[must_use]
    pub fn new(device: &MetalDevice, descriptor: &NDArrayDescriptor) -> Option<Self> {
        // SAFETY: Both pointers come from safe wrappers and are valid for the call.
        let ptr =
            unsafe { ffi::mps_ndarray_new_with_descriptor(device.as_ptr(), descriptor.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps a constructor on `MPSNDArray`.
    #[must_use]
    pub fn scalar(device: &MetalDevice, value: f64) -> Option<Self> {
        // SAFETY: device pointer is valid and we return null or a +1 retained NDArray.
        let ptr = unsafe { ffi::mps_ndarray_new_scalar(device.as_ptr(), value) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps a constructor on `MPSNDArray`.
    #[must_use]
    pub fn new_with_buffer(
        buffer: &MetalBuffer,
        offset: usize,
        descriptor: &NDArrayDescriptor,
    ) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_ndarray_new_with_buffer(buffer.as_ptr(), offset, descriptor.as_ptr())
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps the corresponding `MPSNDArray` method.
    #[must_use]
    pub fn data_type(&self) -> u32 {
        unsafe { ffi::mps_ndarray_data_type(self.ptr) }
    }

    /// Wraps the corresponding `MPSNDArray` method.
    #[must_use]
    pub fn number_of_dimensions(&self) -> usize {
        unsafe { ffi::mps_ndarray_number_of_dimensions(self.ptr) }
    }

    /// Wraps the corresponding `MPSNDArray` method.
    #[must_use]
    pub fn length_of_dimension(&self, dimension_index: usize) -> usize {
        unsafe { ffi::mps_ndarray_length_of_dimension(self.ptr, dimension_index) }
    }

    /// Wraps the corresponding `MPSNDArray` method.
    #[must_use]
    pub fn descriptor(&self) -> Option<NDArrayDescriptor> {
        let ptr = unsafe { ffi::mps_ndarray_descriptor(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(NDArrayDescriptor { ptr })
        }
    }

    /// Wraps the corresponding `MPSNDArray` method.
    #[must_use]
    pub fn resource_size(&self) -> usize {
        unsafe { ffi::mps_ndarray_resource_size(self.ptr) }
    }
}

opaque_handle!(NDArrayIdentity, "Wraps `MPSNDArrayIdentity`.");
impl NDArrayIdentity {
    /// Wraps a constructor on `MPSNDArrayIdentity`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        let ptr = unsafe { ffi::mps_ndarray_identity_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps the corresponding `MPSNDArrayIdentity` method.
    #[must_use]
    pub fn reshape(&self, source: &NDArray, dimension_sizes: &[usize]) -> Option<NDArray> {
        let ptr = unsafe {
            ffi::mps_ndarray_identity_reshape(
                self.ptr,
                ptr::null_mut(),
                source.as_ptr(),
                dimension_sizes.len(),
                dimension_sizes.as_ptr(),
                ptr::null_mut(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(NDArray { ptr })
        }
    }

    /// Wraps the corresponding `MPSNDArrayIdentity` method.
    #[must_use]
    pub fn reshape_with_command_buffer(
        &self,
        command_buffer: &MetalCommandBuffer,
        source: &NDArray,
        dimension_sizes: &[usize],
    ) -> Option<NDArray> {
        let ptr = unsafe {
            ffi::mps_ndarray_identity_reshape(
                self.ptr,
                command_buffer.as_ptr(),
                source.as_ptr(),
                dimension_sizes.len(),
                dimension_sizes.as_ptr(),
                ptr::null_mut(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(NDArray { ptr })
        }
    }

    /// Wraps the corresponding `MPSNDArrayIdentity` method.
    pub fn reshape_into(
        &self,
        command_buffer: Option<&MetalCommandBuffer>,
        source: &NDArray,
        dimension_sizes: &[usize],
        destination: &NDArray,
    ) -> bool {
        let command_buffer_ptr = command_buffer.map_or(ptr::null_mut(), MetalCommandBuffer::as_ptr);
        let ptr = unsafe {
            ffi::mps_ndarray_identity_reshape(
                self.ptr,
                command_buffer_ptr,
                source.as_ptr(),
                dimension_sizes.len(),
                dimension_sizes.as_ptr(),
                destination.as_ptr(),
            )
        };
        !ptr.is_null()
    }
}

opaque_handle!(NDArrayMatrixMultiplication, "Wraps `MPSNDArrayMatrixMultiplication`.");
impl NDArrayMatrixMultiplication {
    /// Wraps a constructor on `MPSNDArrayMatrixMultiplication`.
    #[must_use]
    pub fn new(device: &MetalDevice, source_count: usize) -> Option<Self> {
        let ptr =
            unsafe { ffi::mps_ndarray_matrix_multiplication_new(device.as_ptr(), source_count) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps the corresponding `MPSNDArrayMatrixMultiplication` method.
    #[must_use]
    pub fn alpha(&self) -> f64 {
        unsafe { ffi::mps_ndarray_matrix_multiplication_alpha(self.ptr) }
    }

    /// Wraps the corresponding `MPSNDArrayMatrixMultiplication` setter.
    pub fn set_alpha(&self, alpha: f64) {
        unsafe { ffi::mps_ndarray_matrix_multiplication_set_alpha(self.ptr, alpha) };
    }

    /// Wraps the corresponding `MPSNDArrayMatrixMultiplication` method.
    #[must_use]
    pub fn beta(&self) -> f64 {
        unsafe { ffi::mps_ndarray_matrix_multiplication_beta(self.ptr) }
    }

    /// Wraps the corresponding `MPSNDArrayMatrixMultiplication` setter.
    pub fn set_beta(&self, beta: f64) {
        unsafe { ffi::mps_ndarray_matrix_multiplication_set_beta(self.ptr, beta) };
    }

    /// Wraps the corresponding `MPSNDArrayMatrixMultiplication` encode entry point.
    #[must_use]
    pub fn encode(
        &self,
        command_buffer: &MetalCommandBuffer,
        source_arrays: &[&NDArray],
    ) -> Option<NDArray> {
        let handles: Vec<_> = source_arrays.iter().map(|array| array.as_ptr()).collect();
        let handles_ptr = if handles.is_empty() {
            ptr::null()
        } else {
            handles.as_ptr()
        };
        let ptr = unsafe {
            ffi::mps_ndarray_matrix_multiplication_encode(
                self.ptr,
                command_buffer.as_ptr(),
                source_arrays.len(),
                handles_ptr,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(NDArray { ptr })
        }
    }

    /// Wraps the corresponding `MPSNDArrayMatrixMultiplication` encode entry point.
    pub fn encode_to_destination(
        &self,
        command_buffer: &MetalCommandBuffer,
        source_arrays: &[&NDArray],
        destination: &NDArray,
    ) {
        let handles: Vec<_> = source_arrays.iter().map(|array| array.as_ptr()).collect();
        let handles_ptr = if handles.is_empty() {
            ptr::null()
        } else {
            handles.as_ptr()
        };
        unsafe {
            ffi::mps_ndarray_matrix_multiplication_encode_to_destination(
                self.ptr,
                command_buffer.as_ptr(),
                source_arrays.len(),
                handles_ptr,
                destination.as_ptr(),
            );
        };
    }
}
