use core::ffi::c_void;

extern "C" {
    pub fn mps_unary_encode_image(
        kernel_handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        source_handle: *mut c_void,
        destination_handle: *mut c_void,
    );
    pub fn mps_unary_encode_texture(
        kernel_handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        source_texture_handle: *mut c_void,
        destination_texture_handle: *mut c_void,
    );
    pub fn mps_unary_set_edge_mode(kernel_handle: *mut c_void, edge_mode: usize);
    pub fn mps_unary_set_clip_rect(
        kernel_handle: *mut c_void,
        x: usize,
        y: usize,
        z: usize,
        width: usize,
        height: usize,
        depth: usize,
    );
    pub fn mps_image_scale_set_transform(
        kernel_handle: *mut c_void,
        scale_x: f64,
        scale_y: f64,
        translate_x: f64,
        translate_y: f64,
    );

    pub fn mps_binary_encode_image(
        kernel_handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        primary_handle: *mut c_void,
        secondary_handle: *mut c_void,
        destination_handle: *mut c_void,
    );
    pub fn mps_binary_encode_texture(
        kernel_handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        primary_texture_handle: *mut c_void,
        secondary_texture_handle: *mut c_void,
        destination_texture_handle: *mut c_void,
    );
    pub fn mps_binary_set_primary_edge_mode(kernel_handle: *mut c_void, edge_mode: usize);
    pub fn mps_binary_set_secondary_edge_mode(kernel_handle: *mut c_void, edge_mode: usize);
    pub fn mps_binary_set_clip_rect(
        kernel_handle: *mut c_void,
        x: usize,
        y: usize,
        z: usize,
        width: usize,
        height: usize,
        depth: usize,
    );
    pub fn mps_image_arithmetic_set_scales_bias(
        kernel_handle: *mut c_void,
        primary_scale: f32,
        secondary_scale: f32,
        bias: f32,
    );
    pub fn mps_image_arithmetic_set_clamp(
        kernel_handle: *mut c_void,
        minimum_value: f32,
        maximum_value: f32,
    );

    pub fn mps_image_gaussian_blur_new(device_handle: *mut c_void, sigma: f32) -> *mut c_void;
    pub fn mps_image_box_new(
        device_handle: *mut c_void,
        kernel_width: usize,
        kernel_height: usize,
    ) -> *mut c_void;
    pub fn mps_image_sobel_new(device_handle: *mut c_void, transform: *const f32) -> *mut c_void;
    pub fn mps_image_median_new(device_handle: *mut c_void, kernel_diameter: usize) -> *mut c_void;
    pub fn mps_image_convolution_new(
        device_handle: *mut c_void,
        kernel_width: usize,
        kernel_height: usize,
        weights: *const f32,
    ) -> *mut c_void;
    pub fn mps_image_bilinear_scale_new(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_image_lanczos_scale_new(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_image_threshold_binary_new(
        device_handle: *mut c_void,
        threshold_value: f32,
        maximum_value: f32,
        transform: *const f32,
    ) -> *mut c_void;
    pub fn mps_image_histogram_new(
        device_handle: *mut c_void,
        number_of_entries: usize,
        histogram_for_alpha: bool,
        min_values: *const f32,
        max_values: *const f32,
    ) -> *mut c_void;
    pub fn mps_image_histogram_encode_image(
        histogram_handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        source_handle: *mut c_void,
        histogram_buffer_handle: *mut c_void,
        histogram_offset: usize,
    );
    pub fn mps_image_histogram_encode_texture(
        histogram_handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        source_texture_handle: *mut c_void,
        histogram_buffer_handle: *mut c_void,
        histogram_offset: usize,
    );
    pub fn mps_image_histogram_size_for_source_format(
        histogram_handle: *mut c_void,
        source_format: usize,
    ) -> usize;
    pub fn mps_image_statistics_min_max_new(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_image_statistics_mean_new(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_image_reduce_row_min_new(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_image_reduce_row_max_new(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_image_reduce_row_mean_new(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_image_reduce_row_sum_new(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_image_add_new(device_handle: *mut c_void) -> *mut c_void;
}
