use core::ffi::c_void;

extern "C" {
    pub fn mps_image_new_with_descriptor(
        device_handle: *mut c_void,
        channel_format: usize,
        width: usize,
        height: usize,
        feature_channels: usize,
        number_of_images: usize,
        usage: usize,
        storage_mode: usize,
    ) -> *mut c_void;
    pub fn mps_image_new_with_texture(
        texture_handle: *mut c_void,
        feature_channels: usize,
    ) -> *mut c_void;
    pub fn mps_image_width(handle: *mut c_void) -> usize;
    pub fn mps_image_height(handle: *mut c_void) -> usize;
    pub fn mps_image_feature_channels(handle: *mut c_void) -> usize;
    pub fn mps_image_number_of_images(handle: *mut c_void) -> usize;
    pub fn mps_image_pixel_size(handle: *mut c_void) -> usize;
    pub fn mps_image_pixel_format(handle: *mut c_void) -> usize;
    pub fn mps_get_image_type(handle: *mut c_void) -> u32;
    pub fn mps_image_batch_increment_read_count(
        handles: *const *mut c_void,
        count: usize,
        amount: isize,
    ) -> usize;
    pub fn mps_image_batch_synchronize(
        handles: *const *mut c_void,
        count: usize,
        command_buffer_handle: *mut c_void,
    );
    pub fn mps_image_batch_resource_size(handles: *const *mut c_void, count: usize) -> usize;
    pub fn mps_image_read_bytes(
        handle: *mut c_void,
        data: *mut c_void,
        data_layout: usize,
        bytes_per_row: usize,
        x: usize,
        y: usize,
        z: usize,
        width: usize,
        height: usize,
        depth: usize,
        feature_channel_offset: usize,
        feature_channel_count: usize,
        image_index: usize,
    ) -> bool;
    pub fn mps_image_write_bytes(
        handle: *mut c_void,
        data: *const c_void,
        data_layout: usize,
        bytes_per_row: usize,
        x: usize,
        y: usize,
        z: usize,
        width: usize,
        height: usize,
        depth: usize,
        feature_channel_offset: usize,
        feature_channel_count: usize,
        image_index: usize,
    ) -> bool;
}
