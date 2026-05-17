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

raw_value_type!(MatrixRandomDistribution, usize);
raw_value_type!(MatrixDecompositionStatus, i32);
opaque_generated_handle!(MatrixBatchNormalization);
opaque_generated_handle!(MatrixBatchNormalizationGradient);
opaque_generated_handle!(MatrixBinaryKernel);
opaque_generated_handle!(MatrixCopy);
opaque_generated_handle!(MatrixCopyDescriptor);
opaque_generated_handle!(MatrixCopyToImage);
opaque_generated_handle!(MatrixDecompositionCholesky);
opaque_generated_handle!(MatrixDecompositionLU);
opaque_generated_handle!(MatrixFindTopK);
opaque_generated_handle!(MatrixFullyConnected);
opaque_generated_handle!(MatrixFullyConnectedGradient);
opaque_generated_handle!(MatrixLogSoftMax);
opaque_generated_handle!(MatrixLogSoftMaxGradient);
opaque_generated_handle!(MatrixNeuron);
opaque_generated_handle!(MatrixNeuronGradient);
opaque_generated_handle!(MatrixRandom);
opaque_generated_handle!(MatrixRandomDistributionDescriptor);
opaque_generated_handle!(MatrixRandomMTGP32);
opaque_generated_handle!(MatrixRandomPhilox);
opaque_generated_handle!(MatrixSoftMax);
opaque_generated_handle!(MatrixSoftMaxGradient);
opaque_generated_handle!(MatrixSolveCholesky);
opaque_generated_handle!(MatrixSolveLU);
opaque_generated_handle!(MatrixSolveTriangular);
opaque_generated_handle!(MatrixSum);
opaque_generated_handle!(MatrixUnaryKernel);
opaque_generated_handle!(MatrixVectorMultiplication);
opaque_generated_handle!(TemporaryMatrix);
opaque_generated_handle!(TemporaryVector);
