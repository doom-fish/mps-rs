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

raw_value_type!(AccelerationStructureUsage, usize);
raw_value_type!(RayMaskOptions, usize);
raw_value_type!(TemporalWeighting, usize);
raw_value_type!(TransformType, usize);
raw_value_type!(AccelerationStructureStatus, usize);
raw_value_type!(BoundingBoxIntersectionTestType, usize);
raw_value_type!(PolygonType, usize);
raw_value_type!(RayMaskOperator, usize);
raw_value_type!(TriangleIntersectionTestType, usize);
opaque_generated_handle!(SVGFDefaultTextureAllocator);
opaque_generated_handle!(SVGFDenoiser);
opaque_generated_handle!(TemporalAA);
opaque_generated_handle!(AccelerationStructure);
opaque_generated_handle!(AccelerationStructureGroup);
opaque_generated_handle!(InstanceAccelerationStructure);
opaque_generated_handle!(PolygonBuffer);
opaque_generated_handle!(QuadrilateralAccelerationStructure);
opaque_generated_handle!(TriangleAccelerationStructure);
opaque_generated_handle!(SVGFTextureAllocator);
