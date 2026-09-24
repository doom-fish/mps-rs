use crate::error::{Error, Result};
use crate::ffi;
use apple_metal::{CommandBuffer as MetalCommandBuffer, CommandBufferError, MetalDevice};
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
    (@pointer $name:ident) => {
        impl $name {
            /// Returns the retained Objective-C pointer backing this wrapper.
            #[must_use]
            pub const fn as_ptr(&self) -> *mut c_void {
                self.ptr
            }
        }
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

        opaque_handle!(@pointer $name);
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

/// Wraps `MPSState`.
pub struct State {
    ptr: *mut c_void,
    command_buffer: Option<MetalCommandBuffer>,
}

unsafe impl Send for State {}

impl Drop for State {
    fn drop(&mut self) {
        let ptr = self.ptr;
        if let Some(command_buffer) = &self.command_buffer {
            let _ = guarded(&[command_buffer], &mut || unsafe {
                ffi::mps_state_release(ptr);
            });
        } else {
            unsafe { ffi::mps_object_release(ptr) };
        }
    }
}

opaque_handle!(@pointer State);

impl State {
    /// Wraps a constructor on `MPSState`.
    #[must_use]
    pub fn temporary(command_buffer: &MetalCommandBuffer) -> Option<Self> {
        // SAFETY: command_buffer pointer is valid for the call.
        let ptr = crate::core::encode(command_buffer, |buffer| unsafe {
            ffi::mps_state_temporary_new(buffer)
        })
        .ok()?;
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                command_buffer: Some(command_buffer.clone()),
            })
        }
    }

    /// Wraps a constructor on `MPSState`.
    #[must_use]
    pub fn temporary_with_buffer_size(
        command_buffer: &MetalCommandBuffer,
        buffer_size: usize,
    ) -> Option<Self> {
        let ptr = crate::core::encode(command_buffer, |buffer| unsafe {
            ffi::mps_state_temporary_new_with_buffer_size(buffer, buffer_size)
        })
        .ok()?;
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                command_buffer: Some(command_buffer.clone()),
            })
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
            Some(Self {
                ptr,
                command_buffer: None,
            })
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
            Some(Self {
                ptr,
                command_buffer: None,
            })
        }
    }

    /// Wraps a constructor on `MPSState`.
    #[must_use]
    pub fn temporary_with_resource_list(
        command_buffer: &MetalCommandBuffer,
        resource_list: &StateResourceList,
    ) -> Option<Self> {
        let ptr = crate::core::encode(command_buffer, |buffer| unsafe {
            ffi::mps_state_temporary_new_with_resource_list(buffer, resource_list.as_ptr())
        })
        .ok()?;
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                command_buffer: Some(command_buffer.clone()),
            })
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
    pub fn set_read_count(&self, count: usize) -> Result<()> {
        let Some(command_buffer) = &self.command_buffer else {
            return Err(Error::InvalidArgument(
                "only temporary states have a read count",
            ));
        };
        if self.read_count() == 0 && count > 0 {
            return Err(Error::InvalidArgument(
                "a temporary state whose read count reached zero has returned its storage to MPS",
            ));
        }
        isize::try_from(count).map_err(|_| Error::Overflow)?;
        // SAFETY: self.ptr is a valid State object.
        let accepted = guarded(&[command_buffer], &mut || unsafe {
            ffi::mps_state_set_read_count(self.ptr, count)
        })?;
        accepted
            .then_some(())
            .ok_or(Error::Rejected("MPSState readCount"))
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
                &raw mut width,
                &raw mut height,
                &raw mut depth,
                &raw mut array_length,
                &raw mut pixel_format,
                &raw mut texture_type,
                &raw mut usage,
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
    pub fn synchronize_on_command_buffer(&self, command_buffer: &MetalCommandBuffer) -> Result<()> {
        if self.is_temporary() {
            return Err(Error::InvalidArgument(TEMPORARY_SYNCHRONIZE));
        }
        let accepted = crate::core::encode(command_buffer, |buffer| unsafe {
            ffi::mps_state_synchronize_on_command_buffer(self.ptr, buffer)
        })?;
        accepted
            .then_some(())
            .ok_or(Error::Rejected("MPSState synchronizeOnCommandBuffer"))
    }

    /// Wraps the corresponding `MPSState` method.
    #[must_use]
    pub fn resource_size(&self) -> usize {
        unsafe { ffi::mps_state_resource_size(self.ptr) }
    }
}

fn guarded<R>(command_buffers: &[&MetalCommandBuffer], call: &mut dyn FnMut() -> R) -> Result<R> {
    let Some((command_buffer, rest)) = command_buffers.split_first() else {
        return Ok(call());
    };
    match command_buffer.encode_foreign(|_| guarded(rest, call)) {
        Ok(result) => result,
        Err(CommandBufferError::InvalidState { .. }) => guarded(rest, call),
        Err(error) => Err(Error::CommandBuffer(error)),
    }
}

const TEMPORARY_SYNCHRONIZE: &str =
    "temporary states are GPU-private and cannot be synchronized with the CPU";

/// Calls `MPSStateBatchIncrementReadCount` for the provided `MPSState` values.
pub fn state_batch_increment_read_count(states: &[&State], amount: isize) -> Result<usize> {
    let mut command_buffers: Vec<&MetalCommandBuffer> = Vec::new();
    for command_buffer in states
        .iter()
        .filter_map(|state| state.command_buffer.as_ref())
    {
        if !command_buffers
            .iter()
            .any(|known| known.as_ptr() == command_buffer.as_ptr())
        {
            command_buffers.push(command_buffer);
        }
    }
    let handles: Vec<_> = states.iter().map(|state| state.as_ptr()).collect();
    let handles_ptr = if handles.is_empty() {
        ptr::null()
    } else {
        handles.as_ptr()
    };
    let unique = guarded(&command_buffers, &mut || unsafe {
        ffi::mps_state_batch_increment_read_count(handles_ptr, handles.len(), amount)
    })?;
    usize::try_from(unique).map_err(|_| {
        Error::InvalidArgument("a temporary state's read count would drop below zero or overflow")
    })
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
pub fn state_batch_synchronize(
    states: &[&State],
    command_buffer: &MetalCommandBuffer,
) -> Result<()> {
    if states.iter().any(|state| state.is_temporary()) {
        return Err(Error::InvalidArgument(TEMPORARY_SYNCHRONIZE));
    }
    let handles: Vec<_> = states.iter().map(|state| state.as_ptr()).collect();
    let handles_ptr = if handles.is_empty() {
        ptr::null()
    } else {
        handles.as_ptr()
    };
    let accepted = crate::core::encode(command_buffer, |buffer| unsafe {
        ffi::mps_state_batch_synchronize(handles_ptr, handles.len(), buffer)
    })?;
    accepted
        .then_some(())
        .ok_or(Error::Rejected("MPSStateBatchSynchronize"))
}
