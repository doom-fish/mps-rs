use crate::ffi;
use apple_metal::{CommandBuffer as MetalCommandBuffer, MetalDevice};
use core::ffi::c_void;
use core::ptr;

/// `MPSStateResourceType` constants.
pub mod state_resource_type {
    /// Wraps a `MPSStateResourceType` raw value.
    pub const NONE: usize = 0;
    /// Wraps a `MPSStateResourceType` raw value.
    pub const BUFFER: usize = 1;
    /// Wraps a `MPSStateResourceType` raw value.
    pub const TEXTURE: usize = 2;
}

/// Plain-Rust view of `MPSStateTextureInfo`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StateTextureInfo {
    /// Corresponds to the `width` field on `MPSStateTextureInfo`.
    pub width: usize,
    /// Corresponds to the `height` field on `MPSStateTextureInfo`.
    pub height: usize,
    /// Corresponds to the `depth` field on `MPSStateTextureInfo`.
    pub depth: usize,
    /// Corresponds to the `array_length` field on `MPSStateTextureInfo`.
    pub array_length: usize,
    /// Corresponds to the `pixel_format` field on `MPSStateTextureInfo`.
    pub pixel_format: usize,
    /// Corresponds to the `texture_type` field on `MPSStateTextureInfo`.
    pub texture_type: usize,
    /// Corresponds to the `usage` field on `MPSStateTextureInfo`.
    pub usage: usize,
}

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

opaque_handle!(StateResourceList, "Wraps `MPSStateResourceList`.");
impl StateResourceList {
    /// Wraps a constructor on `MPSStateResourceList`.
    #[must_use]
    pub fn new() -> Option<Self> {
        // SAFETY: This function returns a new StateResourceList or null.
        let ptr = unsafe { ffi::mps_state_resource_list_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps the corresponding `MPSStateResourceList` method.
    pub fn append_buffer(&self, size: usize) {
        // SAFETY: self.ptr is a valid StateResourceList.
        unsafe { ffi::mps_state_resource_list_append_buffer(self.ptr, size) };
    }
}

opaque_handle!(State, "Wraps `MPSState`.");
impl State {
    /// Wraps a constructor on `MPSState`.
    #[must_use]
    pub fn temporary(command_buffer: &MetalCommandBuffer) -> Option<Self> {
        // SAFETY: command_buffer pointer is valid for the call.
        let ptr = unsafe { ffi::mps_state_temporary_new(command_buffer.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps a constructor on `MPSState`.
    #[must_use]
    pub fn temporary_with_buffer_size(
        command_buffer: &MetalCommandBuffer,
        buffer_size: usize,
    ) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_state_temporary_new_with_buffer_size(command_buffer.as_ptr(), buffer_size)
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps a constructor on `MPSState`.
    #[must_use]
    pub fn new_with_buffer_size(device: &MetalDevice, buffer_size: usize) -> Option<Self> {
        // SAFETY: device pointer is valid for the call.
        let ptr = unsafe { ffi::mps_state_new_with_buffer_size(device.as_ptr(), buffer_size) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps a constructor on `MPSState`.
    #[must_use]
    pub fn new_with_resource_list(
        device: &MetalDevice,
        resource_list: &StateResourceList,
    ) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_state_new_with_resource_list(device.as_ptr(), resource_list.as_ptr())
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps a constructor on `MPSState`.
    #[must_use]
    pub fn temporary_with_resource_list(
        command_buffer: &MetalCommandBuffer,
        resource_list: &StateResourceList,
    ) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_state_temporary_new_with_resource_list(
                command_buffer.as_ptr(),
                resource_list.as_ptr(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps the corresponding `MPSState` method.
    #[must_use]
    pub fn resource_count(&self) -> usize {
        // SAFETY: self.ptr is a valid State object.
        unsafe { ffi::mps_state_resource_count(self.ptr) }
    }

    /// Wraps the corresponding `MPSState` method.
    #[must_use]
    pub fn read_count(&self) -> usize {
        // SAFETY: self.ptr is a valid State object.
        unsafe { ffi::mps_state_read_count(self.ptr) }
    }

    /// Wraps the corresponding `MPSState` setter.
    pub fn set_read_count(&self, count: usize) {
        // SAFETY: self.ptr is a valid State object.
        unsafe { ffi::mps_state_set_read_count(self.ptr, count) };
    }

    /// Wraps the corresponding `MPSState` method.
    #[must_use]
    pub fn is_temporary(&self) -> bool {
        unsafe { ffi::mps_state_is_temporary(self.ptr) }
    }

    /// Wraps the corresponding `MPSState` method.
    #[must_use]
    pub fn buffer_size_at_index(&self, index: usize) -> usize {
        unsafe { ffi::mps_state_buffer_size_at_index(self.ptr, index) }
    }

    /// Wraps the corresponding `MPSState` method.
    #[must_use]
    pub fn texture_info_at_index(&self, index: usize) -> StateTextureInfo {
        let mut width = 0;
        let mut height = 0;
        let mut depth = 0;
        let mut array_length = 0;
        let mut pixel_format = 0;
        let mut texture_type = 0;
        let mut usage = 0;
        unsafe {
            ffi::mps_state_texture_info(
                self.ptr,
                index,
                &mut width,
                &mut height,
                &mut depth,
                &mut array_length,
                &mut pixel_format,
                &mut texture_type,
                &mut usage,
            );
        };
        StateTextureInfo {
            width,
            height,
            depth,
            array_length,
            pixel_format,
            texture_type,
            usage,
        }
    }

    /// Wraps the corresponding `MPSState` method.
    #[must_use]
    pub fn resource_type_at_index(&self, index: usize) -> usize {
        unsafe { ffi::mps_state_resource_type_at_index(self.ptr, index) }
    }

    /// Wraps the corresponding `MPSState` method.
    pub fn synchronize_on_command_buffer(&self, command_buffer: &MetalCommandBuffer) {
        unsafe { ffi::mps_state_synchronize_on_command_buffer(self.ptr, command_buffer.as_ptr()) };
    }

    /// Wraps the corresponding `MPSState` method.
    #[must_use]
    pub fn resource_size(&self) -> usize {
        unsafe { ffi::mps_state_resource_size(self.ptr) }
    }
}

/// Calls `MPSStateBatchIncrementReadCount` for the provided `MPSState` values.
#[must_use]
pub fn state_batch_increment_read_count(states: &[&State], amount: isize) -> usize {
    let handles: Vec<_> = states.iter().map(|state| state.as_ptr()).collect();
    let handles_ptr = if handles.is_empty() {
        ptr::null()
    } else {
        handles.as_ptr()
    };
    unsafe { ffi::mps_state_batch_increment_read_count(handles_ptr, handles.len(), amount) }
}

/// Calls `MPSStateBatchResourceSize` for the provided `MPSState` values.
#[must_use]
pub fn state_batch_resource_size(states: &[&State]) -> usize {
    let handles: Vec<_> = states.iter().map(|state| state.as_ptr()).collect();
    let handles_ptr = if handles.is_empty() {
        ptr::null()
    } else {
        handles.as_ptr()
    };
    unsafe { ffi::mps_state_batch_resource_size(handles_ptr, handles.len()) }
}

/// Calls `MPSStateBatchSynchronize` for the provided `MPSState` values.
pub fn state_batch_synchronize(states: &[&State], command_buffer: &MetalCommandBuffer) {
    let handles: Vec<_> = states.iter().map(|state| state.as_ptr()).collect();
    let handles_ptr = if handles.is_empty() {
        ptr::null()
    } else {
        handles.as_ptr()
    };
    unsafe {
        ffi::mps_state_batch_synchronize(handles_ptr, handles.len(), command_buffer.as_ptr());
    };
}
