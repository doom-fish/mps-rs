use crate::ffi;
use crate::image::ImageRegion;
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

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Origin {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Size {
    pub width: f64,
    pub height: f64,
    pub depth: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Region {
    pub origin: Origin,
    pub size: Size,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DimensionSlice {
    pub start: usize,
    pub length: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CustomKernelArgumentCount {
    pub destination_texture_count: usize,
    pub source_texture_count: usize,
    pub broadcast_texture_count: usize,
}

#[must_use]
pub fn rect_no_clip() -> ImageRegion {
    let mut x = 0;
    let mut y = 0;
    let mut z = 0;
    let mut width = 0;
    let mut height = 0;
    let mut depth = 0;
    unsafe { ffi::mps_rect_no_clip(&mut x, &mut y, &mut z, &mut width, &mut height, &mut depth) };
    ImageRegion::new(x, y, z, width, height, depth)
}

raw_value_type!(AliasingStrategy, usize);
raw_value_type!(CustomKernelIndex, usize);
raw_value_type!(DeviceCapsValues, usize);
raw_value_type!(FloatDataTypeBit, usize);
raw_value_type!(FloatDataTypeShift, usize);
opaque_generated_handle!(Kernel);
opaque_generated_handle!(KeyedUnarchiver);
opaque_generated_handle!(DeviceProvider);
opaque_generated_handle!(HeapProvider);
