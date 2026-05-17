use crate::ffi;
use crate::image::Image;
use apple_metal::CommandBuffer as MetalCommandBuffer;
use core::ffi::c_void;
use core::ptr;
use std::collections::HashSet;

macro_rules! opaque_generated_handle {
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

            /// # Safety
            ///
            /// `ptr` must be a valid Objective-C object of the matching MPS type or protocol.
            #[must_use]
            pub unsafe fn retained_from_raw(ptr: *mut c_void) -> Option<Self> {
                let retained = unsafe { ffi::mps_object_retain(ptr) };
                if retained.is_null() {
                    None
                } else {
                    Some(Self { ptr: retained })
                }
            }
        }
    };
}

macro_rules! raw_value_type {
    ($name:ident, $raw:ty) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(pub $raw);

        impl $name {
            #[must_use]
            pub const fn from_raw(raw: $raw) -> Self {
                Self(raw)
            }

            #[must_use]
            pub const fn as_raw(self) -> $raw {
                self.0
            }
        }

        impl From<$raw> for $name {
            fn from(value: $raw) -> Self {
                Self(value)
            }
        }

        impl From<$name> for $raw {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ImageCoordinate {
    pub x: usize,
    pub y: usize,
    pub channel: usize,
}

#[must_use]
pub fn get_image_type(image: &Image) -> ImageType {
    ImageType::from_raw(unsafe { ffi::mps_get_image_type(image.as_ptr()) })
}

#[must_use]
pub fn image_batch_increment_read_count(images: &[&Image], amount: isize) -> usize {
    let handles: Vec<_> = images.iter().map(|image| image.as_ptr()).collect();
    let handles_ptr = if handles.is_empty() {
        ptr::null()
    } else {
        handles.as_ptr()
    };
    unsafe { ffi::mps_image_batch_increment_read_count(handles_ptr, handles.len(), amount) }
}

#[must_use]
pub fn image_batch_resource_size(images: &[&Image]) -> usize {
    let handles: Vec<_> = images.iter().map(|image| image.as_ptr()).collect();
    let handles_ptr = if handles.is_empty() {
        ptr::null()
    } else {
        handles.as_ptr()
    };
    unsafe { ffi::mps_image_batch_resource_size(handles_ptr, handles.len()) }
}

pub fn image_batch_synchronize(images: &[&Image], command_buffer: &MetalCommandBuffer) {
    let handles: Vec<_> = images.iter().map(|image| image.as_ptr()).collect();
    let handles_ptr = if handles.is_empty() {
        ptr::null()
    } else {
        handles.as_ptr()
    };
    unsafe {
        ffi::mps_image_batch_synchronize(handles_ptr, handles.len(), command_buffer.as_ptr());
    }
}

#[must_use]
pub fn image_batch_iterate<F>(images: &[&Image], mut iterator: F) -> isize
where
    F: FnMut(&Image, usize) -> isize,
{
    let mut seen = HashSet::new();
    let mut last = isize::MIN;
    for (index, image) in images.iter().enumerate() {
        if seen.insert(image.as_ptr() as usize) {
            last = iterator(image, index);
            if last > isize::MIN {
                return last;
            }
        }
    }
    last
}

raw_value_type!(AlphaType, usize);
raw_value_type!(ImageType, u32);
raw_value_type!(PurgeableState, usize);
opaque_generated_handle!(BinaryImageKernel);
opaque_generated_handle!(ImageAreaMax);
opaque_generated_handle!(ImageAreaMin);
opaque_generated_handle!(ImageArithmetic);
opaque_generated_handle!(ImageCanny);
opaque_generated_handle!(ImageConversion);
opaque_generated_handle!(ImageCopyToMatrix);
opaque_generated_handle!(ImageDilate);
opaque_generated_handle!(ImageDivide);
opaque_generated_handle!(ImageEDLines);
opaque_generated_handle!(ImageErode);
opaque_generated_handle!(ImageEuclideanDistanceTransform);
opaque_generated_handle!(ImageFindKeypoints);
opaque_generated_handle!(ImageGaussianPyramid);
opaque_generated_handle!(ImageGuidedFilter);
opaque_generated_handle!(ImageHistogramEqualization);
opaque_generated_handle!(ImageHistogramSpecification);
opaque_generated_handle!(ImageIntegral);
opaque_generated_handle!(ImageIntegralOfSquares);
opaque_generated_handle!(ImageLaplacian);
opaque_generated_handle!(ImageLaplacianPyramid);
opaque_generated_handle!(ImageLaplacianPyramidAdd);
opaque_generated_handle!(ImageLaplacianPyramidSubtract);
opaque_generated_handle!(ImageMultiply);
opaque_generated_handle!(ImageNormalizedHistogram);
opaque_generated_handle!(ImagePyramid);
opaque_generated_handle!(ImageReduceColumnMax);
opaque_generated_handle!(ImageReduceColumnMean);
opaque_generated_handle!(ImageReduceColumnMin);
opaque_generated_handle!(ImageReduceColumnSum);
opaque_generated_handle!(ImageReduceUnary);
opaque_generated_handle!(ImageScale);
opaque_generated_handle!(ImageStatisticsMeanAndVariance);
opaque_generated_handle!(ImageSubtract);
opaque_generated_handle!(ImageTent);
opaque_generated_handle!(ImageThresholdBinaryInverse);
opaque_generated_handle!(ImageThresholdToZero);
opaque_generated_handle!(ImageThresholdToZeroInverse);
opaque_generated_handle!(ImageThresholdTruncate);
opaque_generated_handle!(ImageTranspose);
opaque_generated_handle!(TemporaryImage);
opaque_generated_handle!(UnaryImageKernel);
opaque_generated_handle!(ImageAllocator);
opaque_generated_handle!(ImageSizeEncodingState);
opaque_generated_handle!(ImageTransformProvider);
