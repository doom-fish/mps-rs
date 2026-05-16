//! Raw FFI declarations matching the Swift bridge.

#![allow(missing_docs)]

use core::ffi::c_void;

extern "C" {
    pub fn mps_object_release(handle: *mut c_void);

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

    pub fn mps_matrix_descriptor_row_bytes_for_columns(columns: usize, data_type: u32) -> usize;
    pub fn mps_vector_descriptor_vector_bytes_for_length(length: usize, data_type: u32) -> usize;
    pub fn mps_matrix_new_with_buffer(
        buffer_handle: *mut c_void,
        rows: usize,
        columns: usize,
        matrices: usize,
        row_bytes: usize,
        matrix_bytes: usize,
        data_type: u32,
    ) -> *mut c_void;
    pub fn mps_matrix_rows(handle: *mut c_void) -> usize;
    pub fn mps_matrix_columns(handle: *mut c_void) -> usize;
    pub fn mps_matrix_matrices(handle: *mut c_void) -> usize;
    pub fn mps_matrix_row_bytes(handle: *mut c_void) -> usize;
    pub fn mps_matrix_matrix_bytes(handle: *mut c_void) -> usize;
    pub fn mps_matrix_data_type(handle: *mut c_void) -> u32;

    pub fn mps_vector_new_with_buffer(
        buffer_handle: *mut c_void,
        length: usize,
        vectors: usize,
        vector_bytes: usize,
        data_type: u32,
    ) -> *mut c_void;
    pub fn mps_vector_length(handle: *mut c_void) -> usize;
    pub fn mps_vector_vectors(handle: *mut c_void) -> usize;
    pub fn mps_vector_vector_bytes(handle: *mut c_void) -> usize;
    pub fn mps_vector_data_type(handle: *mut c_void) -> u32;

    pub fn mps_matrix_multiplication_new(
        device_handle: *mut c_void,
        transpose_left: bool,
        transpose_right: bool,
        result_rows: usize,
        result_columns: usize,
        interior_columns: usize,
        alpha: f64,
        beta: f64,
    ) -> *mut c_void;
    pub fn mps_matrix_multiplication_encode(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        left_matrix_handle: *mut c_void,
        right_matrix_handle: *mut c_void,
        result_matrix_handle: *mut c_void,
    );
}
