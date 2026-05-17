use crate::ffi;
use core::ffi::c_void;
use core::ptr;

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

raw_value_type!(NDArrayQuantizationScheme, usize);
opaque_generated_handle!(NDArrayAffineInt4Dequantize);
opaque_generated_handle!(NDArrayAffineQuantizationDescriptor);
opaque_generated_handle!(NDArrayBinaryKernel);
opaque_generated_handle!(NDArrayBinaryPrimaryGradientKernel);
opaque_generated_handle!(NDArrayBinarySecondaryGradientKernel);
opaque_generated_handle!(NDArrayGather);
opaque_generated_handle!(NDArrayGatherGradient);
opaque_generated_handle!(NDArrayGatherGradientState);
opaque_generated_handle!(NDArrayGradientState);
opaque_generated_handle!(NDArrayLUTDequantize);
opaque_generated_handle!(NDArrayLUTQuantizationDescriptor);
opaque_generated_handle!(NDArrayMultiaryBase);
opaque_generated_handle!(NDArrayMultiaryGradientKernel);
opaque_generated_handle!(NDArrayMultiaryKernel);
opaque_generated_handle!(NDArrayQuantizationDescriptor);
opaque_generated_handle!(NDArrayQuantizedMatrixMultiplication);
opaque_generated_handle!(NDArrayStridedSlice);
opaque_generated_handle!(NDArrayStridedSliceGradient);
opaque_generated_handle!(NDArrayUnaryGradientKernel);
opaque_generated_handle!(NDArrayUnaryKernel);
opaque_generated_handle!(NDArrayVectorLUTDequantize);
opaque_generated_handle!(TemporaryNDArray);
opaque_generated_handle!(NDArrayAllocator);
