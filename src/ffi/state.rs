use core::ffi::c_void;

extern "C" {
    pub fn mps_state_resource_list_new() -> *mut c_void;
    pub fn mps_state_resource_list_append_buffer(handle: *mut c_void, size: usize);

    pub fn mps_state_temporary_new(command_buffer_handle: *mut c_void) -> *mut c_void;
    pub fn mps_state_temporary_new_with_buffer_size(
        command_buffer_handle: *mut c_void,
        buffer_size: usize,
    ) -> *mut c_void;
    pub fn mps_state_new_with_buffer_size(device_handle: *mut c_void, buffer_size: usize)
        -> *mut c_void;
    pub fn mps_state_new_with_resource_list(
        device_handle: *mut c_void,
        resource_list_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mps_state_temporary_new_with_resource_list(
        command_buffer_handle: *mut c_void,
        resource_list_handle: *mut c_void,
    ) -> *mut c_void;

    pub fn mps_state_resource_count(handle: *mut c_void) -> usize;
    pub fn mps_state_read_count(handle: *mut c_void) -> usize;
    pub fn mps_state_set_read_count(handle: *mut c_void, value: usize);
    pub fn mps_state_is_temporary(handle: *mut c_void) -> bool;
    pub fn mps_state_buffer_size_at_index(handle: *mut c_void, index: usize) -> usize;
    pub fn mps_state_texture_info(
        handle: *mut c_void,
        index: usize,
        width: *mut usize,
        height: *mut usize,
        depth: *mut usize,
        array_length: *mut usize,
        pixel_format: *mut usize,
        texture_type: *mut usize,
        usage: *mut usize,
    );
    pub fn mps_state_resource_type_at_index(handle: *mut c_void, index: usize) -> usize;
    pub fn mps_state_synchronize_on_command_buffer(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
    );
    pub fn mps_state_resource_size(handle: *mut c_void) -> usize;

    pub fn mps_state_batch_increment_read_count(
        handles: *const *mut c_void,
        count: usize,
        amount: isize,
    ) -> usize;
    pub fn mps_state_batch_synchronize(
        handles: *const *mut c_void,
        count: usize,
        command_buffer_handle: *mut c_void,
    );
    pub fn mps_state_batch_resource_size(handles: *const *mut c_void, count: usize) -> usize;
}
