use core::ffi::c_void;

extern "C" {
    pub fn mps_nn_image_node_new() -> *mut c_void;
    pub fn mps_nn_image_node_exported() -> *mut c_void;
    pub fn mps_nn_image_node_format(handle: *mut c_void) -> usize;
    pub fn mps_nn_image_node_set_format(handle: *mut c_void, format: usize);
    pub fn mps_nn_image_node_export_from_graph(handle: *mut c_void) -> bool;
    pub fn mps_nn_image_node_set_export_from_graph(handle: *mut c_void, export: bool);
    pub fn mps_nn_image_node_synchronize_resource(handle: *mut c_void) -> bool;
    pub fn mps_nn_image_node_set_synchronize_resource(handle: *mut c_void, synchronize: bool);
    pub fn mps_nn_image_node_use_default_allocator(handle: *mut c_void);

    pub fn mps_cnn_neuron_relu_node_new(source_handle: *mut c_void, a: f32) -> *mut c_void;
    pub fn mps_cnn_pooling_max_node_new(
        source_handle: *mut c_void,
        filter_size: usize,
        stride: usize,
    ) -> *mut c_void;
    pub fn mps_cnn_softmax_node_new(source_handle: *mut c_void) -> *mut c_void;
    pub fn mps_cnn_upsampling_nearest_node_new(
        source_handle: *mut c_void,
        scale_x: usize,
        scale_y: usize,
    ) -> *mut c_void;
    pub fn mps_nn_filter_node_result_image(handle: *mut c_void) -> *mut c_void;

    pub fn mps_nn_graph_new(
        device_handle: *mut c_void,
        result_image_handle: *mut c_void,
        result_image_is_needed: bool,
    ) -> *mut c_void;
    pub fn mps_nn_graph_source_image_count(handle: *mut c_void) -> usize;
    pub fn mps_nn_graph_format(handle: *mut c_void) -> usize;
    pub fn mps_nn_graph_set_format(handle: *mut c_void, format: usize);
    pub fn mps_nn_graph_set_output_state_is_temporary(handle: *mut c_void, temporary: bool);
    pub fn mps_nn_graph_use_default_destination_image_allocator(handle: *mut c_void);
    pub fn mps_nn_graph_reload_from_data_sources(handle: *mut c_void);
    pub fn mps_nn_graph_encode(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        source_image_count: usize,
        source_image_handles: *const *mut c_void,
    ) -> *mut c_void;

    pub fn mps_cnn_convolution_descriptor_new(
        kernel_width: usize,
        kernel_height: usize,
        input_feature_channels: usize,
        output_feature_channels: usize,
    ) -> *mut c_void;
    pub fn mps_cnn_convolution_descriptor_kernel_width(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_descriptor_kernel_height(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_descriptor_stride_in_pixels_x(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_descriptor_set_stride_in_pixels_x(handle: *mut c_void, value: usize);
    pub fn mps_cnn_convolution_descriptor_stride_in_pixels_y(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_descriptor_set_stride_in_pixels_y(handle: *mut c_void, value: usize);
    pub fn mps_cnn_convolution_descriptor_groups(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_descriptor_set_groups(handle: *mut c_void, value: usize);
    pub fn mps_cnn_convolution_descriptor_dilation_rate_x(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_descriptor_set_dilation_rate_x(handle: *mut c_void, value: usize);
    pub fn mps_cnn_convolution_descriptor_dilation_rate_y(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_descriptor_set_dilation_rate_y(handle: *mut c_void, value: usize);

    pub fn mps_rnn_single_gate_descriptor_new(
        input_feature_channels: usize,
        output_feature_channels: usize,
    ) -> *mut c_void;
    pub fn mps_rnn_single_gate_descriptor_input_feature_channels(handle: *mut c_void) -> usize;
    pub fn mps_rnn_single_gate_descriptor_set_input_feature_channels(
        handle: *mut c_void,
        value: usize,
    );
    pub fn mps_rnn_single_gate_descriptor_output_feature_channels(handle: *mut c_void) -> usize;
    pub fn mps_rnn_single_gate_descriptor_set_output_feature_channels(
        handle: *mut c_void,
        value: usize,
    );
    pub fn mps_rnn_single_gate_descriptor_use_layer_input_unit_transform_mode(
        handle: *mut c_void,
    ) -> bool;
    pub fn mps_rnn_single_gate_descriptor_set_use_layer_input_unit_transform_mode(
        handle: *mut c_void,
        value: bool,
    );
    pub fn mps_rnn_single_gate_descriptor_use_float32_weights(handle: *mut c_void) -> bool;
    pub fn mps_rnn_single_gate_descriptor_set_use_float32_weights(handle: *mut c_void, value: bool);
    pub fn mps_rnn_single_gate_descriptor_layer_sequence_direction(handle: *mut c_void) -> usize;
    pub fn mps_rnn_single_gate_descriptor_set_layer_sequence_direction(
        handle: *mut c_void,
        value: usize,
    );
}
