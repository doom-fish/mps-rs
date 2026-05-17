use crate::ffi;
use apple_metal::{
    CommandBuffer as MetalCommandBuffer, CommandQueue, ManuallyDropDevice, MetalBuffer, MetalDevice,
};
use core::ffi::c_void;
use core::ptr;

pub mod device_options {
    pub const DEFAULT: usize = 0;
    pub const LOW_POWER: usize = 1;
    pub const SKIP_REMOVABLE: usize = 2;
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
                    // SAFETY: `ptr` is a +1 retained MPS object owned by this wrapper.
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

pub fn supports_mtl_device(device: &MetalDevice) -> bool {
    // SAFETY: The device pointer is valid for the call and we just read the return value.
    unsafe { ffi::mps_supports_mtl_device(device.as_ptr()) }
}

opaque_handle!(PreferredDevice);
impl PreferredDevice {
    #[must_use]
    pub fn as_borrowed_device(&self) -> ManuallyDropDevice {
        // SAFETY: The device pointer is a valid MTLDevice reference held by this wrapper.
        unsafe { MetalDevice::from_raw_borrowed(self.ptr) }
    }
}

#[must_use]
pub fn preferred_device(options: usize) -> Option<PreferredDevice> {
    // SAFETY: This function returns a +1 retained MTLDevice or null.
    let ptr = unsafe { ffi::mps_get_preferred_device(options) };
    if ptr.is_null() {
        None
    } else {
        Some(PreferredDevice { ptr })
    }
}

pub fn hint_temporary_memory_high_water_mark(command_buffer: &MetalCommandBuffer, bytes: usize) {
    // SAFETY: The command buffer pointer is valid for the call.
    unsafe { ffi::mps_hint_temporary_memory_high_water_mark(command_buffer.as_ptr(), bytes) };
}

pub use crate::generated::core::*;

pub fn set_heap_cache_duration(command_buffer: &MetalCommandBuffer, seconds: f64) {
    // SAFETY: The command buffer pointer is valid for the call.
    unsafe { ffi::mps_set_heap_cache_duration(command_buffer.as_ptr(), seconds) };
}

opaque_handle!(Predicate);
impl Predicate {
    #[must_use]
    pub fn new_with_buffer(buffer: &MetalBuffer, offset: usize) -> Option<Self> {
        // SAFETY: This function returns a +1 retained predicate or null.
        let ptr = unsafe { ffi::mps_predicate_new_with_buffer(buffer.as_ptr(), offset) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn new_with_device(device: &MetalDevice) -> Option<Self> {
        // SAFETY: This function returns a +1 retained predicate or null.
        let ptr = unsafe { ffi::mps_predicate_new_with_device(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn predicate_offset(&self) -> usize {
        // SAFETY: The predicate pointer is valid for the call.
        unsafe { ffi::mps_predicate_offset(self.ptr) }
    }
}

opaque_handle!(CommandBuffer);
impl CommandBuffer {
    #[must_use]
    pub fn new_with_command_buffer(command_buffer: &MetalCommandBuffer) -> Option<Self> {
        // SAFETY: This function returns a +1 retained command buffer or null.
        let ptr =
            unsafe { ffi::mps_command_buffer_new_with_command_buffer(command_buffer.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn from_command_queue(command_queue: &CommandQueue) -> Option<Self> {
        // SAFETY: This function returns a +1 retained command buffer or null.
        let ptr = unsafe { ffi::mps_command_buffer_from_command_queue(command_queue.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    pub fn set_predicate(&self, predicate: &Predicate) {
        // SAFETY: Both pointers are valid for the call.
        unsafe { ffi::mps_command_buffer_set_predicate(self.ptr, predicate.as_ptr()) };
    }

    pub fn clear_predicate(&self) {
        // SAFETY: The command buffer pointer is valid for the call.
        unsafe { ffi::mps_command_buffer_clear_predicate(self.ptr) };
    }

    pub fn prefetch_heap_for_workload_size(&self, size: usize) {
        // SAFETY: The command buffer pointer is valid for the call.
        unsafe { ffi::mps_command_buffer_prefetch_heap(self.ptr, size) };
    }

    pub fn commit_and_continue(&self) {
        // SAFETY: The command buffer pointer is valid for the call.
        unsafe { ffi::mps_command_buffer_commit_and_continue(self.ptr) };
    }
}
