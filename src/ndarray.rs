use crate::error::{Error, Result};
use crate::ffi;
use crate::matrix::data_type_size;
use apple_metal::{CommandBuffer as MetalCommandBuffer, MetalBuffer, MetalDevice};
use core::ffi::c_void;
use core::ptr;

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
        validate_dimension_sizes(dimension_sizes).ok()?;
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
    pub fn set_number_of_dimensions(&self, number_of_dimensions: usize) -> Result<()> {
        if number_of_dimensions > MAX_DIMENSIONS {
            return Err(Error::InvalidArgument(
                "an NDArray has at most 16 dimensions",
            ));
        }
        // SAFETY: self.ptr is a valid NDArrayDescriptor.
        unsafe {
            ffi::mps_ndarray_descriptor_set_number_of_dimensions(self.ptr, number_of_dimensions);
        };
        Ok(())
    }

    /// Wraps the corresponding `MPSNDArrayDescriptor` method.
    #[must_use]
    pub fn length_of_dimension(&self, dimension_index: usize) -> usize {
        // SAFETY: self.ptr is a valid NDArrayDescriptor and dimension_index is in bounds.
        unsafe { ffi::mps_ndarray_descriptor_length_of_dimension(self.ptr, dimension_index) }
    }

    /// Wraps the corresponding `MPSNDArrayDescriptor` method.
    pub fn reshape_with_dimension_sizes(&self, dimension_sizes: &[usize]) -> Result<()> {
        validate_dimension_sizes(dimension_sizes)?;
        // SAFETY: dimension_sizes.as_ptr() is valid for dimension_sizes.len() elements.
        unsafe {
            ffi::mps_ndarray_descriptor_reshape_with_dimension_sizes(
                self.ptr,
                dimension_sizes.len(),
                dimension_sizes.as_ptr(),
            );
        };
        Ok(())
    }

    /// Wraps the corresponding `MPSNDArrayDescriptor` method.
    pub fn transpose_dimension(
        &self,
        dimension_index: usize,
        other_dimension_index: usize,
    ) -> Result<()> {
        let dimensions = self.number_of_dimensions();
        if dimension_index >= dimensions || other_dimension_index >= dimensions {
            return Err(Error::InvalidArgument(
                "transposed dimensions must be below number_of_dimensions",
            ));
        }
        // SAFETY: Both dimension indices are below the descriptor's rank.
        unsafe {
            ffi::mps_ndarray_descriptor_transpose_dimension(
                self.ptr,
                dimension_index,
                other_dimension_index,
            );
        };
        Ok(())
    }

    fn lengths(&self) -> Vec<usize> {
        (0..self.number_of_dimensions())
            .map(|dimension| self.length_of_dimension(dimension))
            .collect()
    }

    pub fn required_buffer_length(&self) -> Result<usize> {
        let data_type = self.data_type();
        let element_size =
            data_type_size(data_type).ok_or(Error::UnsupportedDataType(data_type))?;
        let lengths = self.lengths();
        let row = lengths
            .first()
            .copied()
            .unwrap_or(1)
            .checked_mul(element_size)
            .and_then(|bytes| bytes.checked_next_multiple_of(ROW_ALIGNMENT))
            .ok_or(Error::Overflow)?;
        lengths
            .iter()
            .skip(1)
            .try_fold(row, |bytes, length| bytes.checked_mul(*length))
            .filter(|bytes| isize::try_from(*bytes).is_ok())
            .ok_or(Error::Overflow)
    }
}

const MAX_DIMENSIONS: usize = 16;
const MAX_ELEMENTS: usize = (1 << 31) - 1;
const ROW_ALIGNMENT: usize = 16;

fn volume(lengths: impl IntoIterator<Item = usize>) -> Option<usize> {
    lengths.into_iter().try_fold(1_usize, usize::checked_mul)
}

fn validate_dimension_sizes(dimension_sizes: &[usize]) -> Result<()> {
    if dimension_sizes.is_empty() || dimension_sizes.len() > MAX_DIMENSIONS {
        return Err(Error::InvalidArgument(
            "an NDArray has between 1 and 16 dimensions",
        ));
    }
    volume(dimension_sizes.iter().copied())
        .filter(|elements| *elements <= MAX_ELEMENTS)
        .map(|_| ())
        .ok_or(Error::InvalidArgument(
            "an NDArray holds fewer than 2^31 elements",
        ))
}

opaque_handle!(NDArray, "Wraps `MPSNDArray`.", sync);
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
    pub fn new_with_buffer(
        buffer: &MetalBuffer,
        offset: usize,
        descriptor: &NDArrayDescriptor,
    ) -> Result<Self> {
        if !unsafe { ffi::mps_ndarray_buffer_backing_available() } {
            return Err(Error::Unsupported(
                "buffer-backed NDArrays need macOS 15 or later",
            ));
        }
        let data_type = descriptor.data_type();
        let element_size =
            data_type_size(data_type).ok_or(Error::UnsupportedDataType(data_type))?;
        if offset % element_size != 0 {
            return Err(Error::Misaligned {
                field: "offset",
                value: offset,
                alignment: element_size,
            });
        }
        let required = offset
            .checked_add(descriptor.required_buffer_length()?)
            .ok_or(Error::Overflow)?;
        let length = buffer.length();
        if required > length {
            return Err(Error::BufferTooSmall { required, length });
        }
        let ptr = unsafe {
            ffi::mps_ndarray_new_with_buffer(buffer.as_ptr(), offset, descriptor.as_ptr())
        };
        if ptr.is_null() {
            Err(Error::Rejected("MPSNDArray initWithBuffer"))
        } else {
            Ok(Self { ptr })
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

    fn lengths(&self) -> Vec<usize> {
        (0..self.number_of_dimensions())
            .map(|dimension| self.length_of_dimension(dimension))
            .collect()
    }
}

fn reshape_is_valid(
    source: &NDArray,
    dimension_sizes: &[usize],
    destination: Option<&NDArray>,
) -> bool {
    if dimension_sizes.is_empty() || dimension_sizes.len() > MAX_DIMENSIONS {
        return false;
    }
    let target = volume(dimension_sizes.iter().copied());
    if target.is_none() || target != volume(source.lengths()) {
        return false;
    }
    destination.is_none_or(|destination| {
        destination.number_of_dimensions() == dimension_sizes.len()
            && dimension_sizes
                .iter()
                .rev()
                .enumerate()
                .all(|(dimension, size)| destination.length_of_dimension(dimension) == *size)
    })
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
        if !reshape_is_valid(source, dimension_sizes, None) {
            return None;
        }
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
        crate::core::ensure_recording(command_buffer).ok()?;
        if !reshape_is_valid(source, dimension_sizes, None) {
            return None;
        }
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
        if command_buffer.is_some_and(|buffer| crate::core::ensure_recording(buffer).is_err())
            || !reshape_is_valid(source, dimension_sizes, Some(destination))
        {
            return false;
        }
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

/// Wraps `MPSNDArrayMatrixMultiplication`.
pub struct NDArrayMatrixMultiplication {
    ptr: *mut c_void,
    source_count: usize,
}

unsafe impl Send for NDArrayMatrixMultiplication {}

impl Drop for NDArrayMatrixMultiplication {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` is a +1 retained MPS object owned by this wrapper.
            unsafe { ffi::mps_object_release(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl NDArrayMatrixMultiplication {
    /// Wraps a constructor on `MPSNDArrayMatrixMultiplication`.
    #[must_use]
    pub fn new(device: &MetalDevice, source_count: usize) -> Option<Self> {
        if !(2..=3).contains(&source_count) {
            return None;
        }
        let ptr =
            unsafe { ffi::mps_ndarray_matrix_multiplication_new(device.as_ptr(), source_count) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr, source_count })
        }
    }

    /// Returns the retained Objective-C pointer backing this wrapper.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    #[must_use]
    pub const fn source_count(&self) -> usize {
        self.source_count
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
        crate::core::ensure_recording(command_buffer).ok()?;
        validate_multiplication(self.source_count, source_arrays, None).ok()?;
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
    ) -> Result<()> {
        crate::core::ensure_recording(command_buffer)?;
        validate_multiplication(self.source_count, source_arrays, Some(destination))?;
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
        Ok(())
    }
}

fn validate_multiplication(
    source_count: usize,
    sources: &[&NDArray],
    destination: Option<&NDArray>,
) -> Result<()> {
    if sources.len() != source_count {
        return Err(Error::DimensionMismatch {
            field: "source array count",
            expected: source_count,
            actual: sources.len(),
        });
    }
    let (left, right) = (sources[0], sources[1]);
    let interior = left.length_of_dimension(0);
    if right.length_of_dimension(1) != interior {
        return Err(Error::DimensionMismatch {
            field: "interior dimension",
            expected: interior,
            actual: right.length_of_dimension(1),
        });
    }
    let rank = sources
        .iter()
        .copied()
        .chain(destination)
        .map(NDArray::number_of_dimensions)
        .fold(2, usize::max);
    let mut result = vec![right.length_of_dimension(0), left.length_of_dimension(1)];
    for dimension in 2..rank {
        let (a, b) = (
            left.length_of_dimension(dimension),
            right.length_of_dimension(dimension),
        );
        if a != b && a != 1 && b != 1 {
            return Err(Error::DimensionMismatch {
                field: "batch dimension",
                expected: a,
                actual: b,
            });
        }
        result.push(a.max(b));
    }
    let covers = |array: &NDArray, field: &'static str, broadcast: bool| {
        result
            .iter()
            .enumerate()
            .try_for_each(|(dimension, expected)| {
                let actual = array.length_of_dimension(dimension);
                if actual >= *expected || (broadcast && actual == 1) {
                    Ok(())
                } else {
                    Err(Error::DimensionMismatch {
                        field,
                        expected: *expected,
                        actual,
                    })
                }
            })
    };
    if let Some(addend) = sources.get(2) {
        covers(addend, "addend dimension", true)?;
    }
    if let Some(destination) = destination {
        covers(destination, "destination dimension", false)?;
    }
    Ok(())
}
