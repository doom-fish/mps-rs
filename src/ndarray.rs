use crate::ffi;
use apple_metal::{CommandBuffer as MetalCommandBuffer, MetalBuffer, MetalDevice};
use core::ffi::c_void;
use core::ptr;

macro_rules! opaque_handle {
    ($name:ident) => {
        pub struct $name {
            ptr: *mut c_void,
        }

        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}

        impl Drop for $name {
            fn drop(&mut self) {
                if !self.ptr.is_null() {
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

opaque_handle!(NDArrayDescriptor);
impl NDArrayDescriptor {
    #[must_use]
    pub fn with_dimension_sizes(data_type: u32, dimension_sizes: &[usize]) -> Option<Self> {
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

    #[must_use]
    pub fn data_type(&self) -> u32 {
        unsafe { ffi::mps_ndarray_descriptor_data_type(self.ptr) }
    }

    pub fn set_data_type(&self, data_type: u32) {
        unsafe { ffi::mps_ndarray_descriptor_set_data_type(self.ptr, data_type) };
    }

    #[must_use]
    pub fn number_of_dimensions(&self) -> usize {
        unsafe { ffi::mps_ndarray_descriptor_number_of_dimensions(self.ptr) }
    }

    pub fn set_number_of_dimensions(&self, number_of_dimensions: usize) {
        unsafe {
            ffi::mps_ndarray_descriptor_set_number_of_dimensions(self.ptr, number_of_dimensions);
        };
    }

    #[must_use]
    pub fn length_of_dimension(&self, dimension_index: usize) -> usize {
        unsafe { ffi::mps_ndarray_descriptor_length_of_dimension(self.ptr, dimension_index) }
    }

    pub fn reshape_with_dimension_sizes(&self, dimension_sizes: &[usize]) {
        unsafe {
            ffi::mps_ndarray_descriptor_reshape_with_dimension_sizes(
                self.ptr,
                dimension_sizes.len(),
                dimension_sizes.as_ptr(),
            );
        };
    }

    pub fn transpose_dimension(&self, dimension_index: usize, other_dimension_index: usize) {
        unsafe {
            ffi::mps_ndarray_descriptor_transpose_dimension(
                self.ptr,
                dimension_index,
                other_dimension_index,
            );
        };
    }
}

opaque_handle!(NDArray);
impl NDArray {
    #[must_use]
    pub fn new(device: &MetalDevice, descriptor: &NDArrayDescriptor) -> Option<Self> {
        let ptr =
            unsafe { ffi::mps_ndarray_new_with_descriptor(device.as_ptr(), descriptor.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn scalar(device: &MetalDevice, value: f64) -> Option<Self> {
        let ptr = unsafe { ffi::mps_ndarray_new_scalar(device.as_ptr(), value) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

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

    #[must_use]
    pub fn data_type(&self) -> u32 {
        unsafe { ffi::mps_ndarray_data_type(self.ptr) }
    }

    #[must_use]
    pub fn number_of_dimensions(&self) -> usize {
        unsafe { ffi::mps_ndarray_number_of_dimensions(self.ptr) }
    }

    #[must_use]
    pub fn length_of_dimension(&self, dimension_index: usize) -> usize {
        unsafe { ffi::mps_ndarray_length_of_dimension(self.ptr, dimension_index) }
    }

    #[must_use]
    pub fn descriptor(&self) -> Option<NDArrayDescriptor> {
        let ptr = unsafe { ffi::mps_ndarray_descriptor(self.ptr) };
        if ptr.is_null() {
            None
        } else {
            Some(NDArrayDescriptor { ptr })
        }
    }

    #[must_use]
    pub fn resource_size(&self) -> usize {
        unsafe { ffi::mps_ndarray_resource_size(self.ptr) }
    }
}

opaque_handle!(NDArrayIdentity);
impl NDArrayIdentity {
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        let ptr = unsafe { ffi::mps_ndarray_identity_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

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

opaque_handle!(NDArrayMatrixMultiplication);
impl NDArrayMatrixMultiplication {
    #[must_use]
    pub fn new(device: &MetalDevice, source_count: usize) -> Option<Self> {
        let ptr = unsafe { ffi::mps_ndarray_matrix_multiplication_new(device.as_ptr(), source_count) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn alpha(&self) -> f64 {
        unsafe { ffi::mps_ndarray_matrix_multiplication_alpha(self.ptr) }
    }

    pub fn set_alpha(&self, alpha: f64) {
        unsafe { ffi::mps_ndarray_matrix_multiplication_set_alpha(self.ptr, alpha) };
    }

    #[must_use]
    pub fn beta(&self) -> f64 {
        unsafe { ffi::mps_ndarray_matrix_multiplication_beta(self.ptr) }
    }

    pub fn set_beta(&self, beta: f64) {
        unsafe { ffi::mps_ndarray_matrix_multiplication_set_beta(self.ptr, beta) };
    }

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
