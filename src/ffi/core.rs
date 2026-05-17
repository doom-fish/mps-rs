use core::ffi::c_void;

extern "C" {
    pub fn mps_object_retain(handle: *mut c_void) -> *mut c_void;
    pub fn mps_object_release(handle: *mut c_void);

    pub fn mps_supports_mtl_device(device_handle: *mut c_void) -> bool;
    pub fn mps_get_preferred_device(options: usize) -> *mut c_void;
    pub fn mps_hint_temporary_memory_high_water_mark(
        command_buffer_handle: *mut c_void,
        bytes: usize,
    );
    pub fn mps_set_heap_cache_duration(command_buffer_handle: *mut c_void, seconds: f64);
    pub fn mps_rect_no_clip(
        x: *mut usize,
        y: *mut usize,
        z: *mut usize,
        width: *mut usize,
        height: *mut usize,
        depth: *mut usize,
    );

    pub fn mps_predicate_new_with_buffer(buffer_handle: *mut c_void, offset: usize) -> *mut c_void;
    pub fn mps_predicate_new_with_device(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_predicate_offset(handle: *mut c_void) -> usize;

    pub fn mps_command_buffer_new_with_command_buffer(
        command_buffer_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mps_command_buffer_from_command_queue(command_queue_handle: *mut c_void) -> *mut c_void;
    pub fn mps_command_buffer_set_predicate(
        command_buffer_handle: *mut c_void,
        predicate_handle: *mut c_void,
    );
    pub fn mps_command_buffer_clear_predicate(command_buffer_handle: *mut c_void);
    pub fn mps_command_buffer_prefetch_heap(command_buffer_handle: *mut c_void, size: usize);
    pub fn mps_command_buffer_commit_and_continue(command_buffer_handle: *mut c_void);
}
