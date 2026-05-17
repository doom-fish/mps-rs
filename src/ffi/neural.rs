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

    pub fn mps_cnn_convolution_descriptor_input_feature_channels(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_descriptor_output_feature_channels(handle: *mut c_void) -> usize;

    pub fn mps_cnn_convolution_new(
        device_handle: *mut c_void,
        descriptor_handle: *mut c_void,
        kernel_weights: *const f32,
        bias_terms: *const f32,
        flags: usize,
    ) -> *mut c_void;
    pub fn mps_cnn_convolution_input_feature_channels(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_output_feature_channels(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_groups(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_sub_pixel_scale_factor(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_channel_multiplier(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_accumulator_precision_option(handle: *mut c_void) -> usize;
    pub fn mps_cnn_convolution_set_accumulator_precision_option(handle: *mut c_void, value: usize);
    pub fn mps_cnn_convolution_encode_image(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        source_image_handle: *mut c_void,
        destination_image_handle: *mut c_void,
    );

    pub fn mps_cnn_convolution_weights_and_biases_state_new(
        weights_handle: *mut c_void,
        biases_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mps_cnn_convolution_weights_and_biases_state_new_with_offsets(
        weights_handle: *mut c_void,
        weights_offset: usize,
        biases_handle: *mut c_void,
        biases_offset: usize,
        descriptor_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mps_cnn_convolution_weights_and_biases_state_new_with_device(
        device_handle: *mut c_void,
        descriptor_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mps_cnn_convolution_weights_and_biases_state_weights_offset(
        handle: *mut c_void,
    ) -> usize;
    pub fn mps_cnn_convolution_weights_and_biases_state_biases_offset(handle: *mut c_void)
        -> usize;

    pub fn mps_nn_optimizer_descriptor_new(
        learning_rate: f32,
        gradient_rescale: f32,
        regularization_type: usize,
        regularization_scale: f32,
    ) -> *mut c_void;
    pub fn mps_nn_optimizer_descriptor_new_with_gradient_clipping(
        learning_rate: f32,
        gradient_rescale: f32,
        apply_gradient_clipping: bool,
        gradient_clip_max: f32,
        gradient_clip_min: f32,
        regularization_type: usize,
        regularization_scale: f32,
    ) -> *mut c_void;
    pub fn mps_nn_optimizer_descriptor_learning_rate(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_descriptor_set_learning_rate(handle: *mut c_void, value: f32);
    pub fn mps_nn_optimizer_descriptor_gradient_rescale(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_descriptor_set_gradient_rescale(handle: *mut c_void, value: f32);
    pub fn mps_nn_optimizer_descriptor_apply_gradient_clipping(handle: *mut c_void) -> bool;
    pub fn mps_nn_optimizer_descriptor_set_apply_gradient_clipping(
        handle: *mut c_void,
        value: bool,
    );
    pub fn mps_nn_optimizer_descriptor_gradient_clip_max(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_descriptor_set_gradient_clip_max(handle: *mut c_void, value: f32);
    pub fn mps_nn_optimizer_descriptor_gradient_clip_min(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_descriptor_set_gradient_clip_min(handle: *mut c_void, value: f32);
    pub fn mps_nn_optimizer_descriptor_regularization_scale(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_descriptor_set_regularization_scale(handle: *mut c_void, value: f32);
    pub fn mps_nn_optimizer_descriptor_regularization_type(handle: *mut c_void) -> usize;
    pub fn mps_nn_optimizer_descriptor_set_regularization_type(handle: *mut c_void, value: usize);

    pub fn mps_nn_optimizer_learning_rate(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_set_learning_rate(handle: *mut c_void, value: f32);
    pub fn mps_nn_optimizer_gradient_rescale(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_apply_gradient_clipping(handle: *mut c_void) -> bool;
    pub fn mps_nn_optimizer_set_apply_gradient_clipping(handle: *mut c_void, value: bool);
    pub fn mps_nn_optimizer_gradient_clip_max(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_gradient_clip_min(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_regularization_scale(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_regularization_type(handle: *mut c_void) -> usize;

    pub fn mps_nn_optimizer_sgd_new(device_handle: *mut c_void, learning_rate: f32) -> *mut c_void;
    pub fn mps_nn_optimizer_sgd_new_with_options(
        device_handle: *mut c_void,
        momentum_scale: f32,
        use_nesterov_momentum: bool,
        optimizer_descriptor_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mps_nn_optimizer_sgd_momentum_scale(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_sgd_use_nesterov_momentum(handle: *mut c_void) -> bool;
    pub fn mps_nn_optimizer_sgd_encode_vector(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        input_gradient_vector_handle: *mut c_void,
        input_values_vector_handle: *mut c_void,
        input_momentum_vector_handle: *mut c_void,
        result_values_vector_handle: *mut c_void,
    );
    pub fn mps_nn_optimizer_sgd_encode_matrix(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        input_gradient_matrix_handle: *mut c_void,
        input_values_matrix_handle: *mut c_void,
        input_momentum_matrix_handle: *mut c_void,
        result_values_matrix_handle: *mut c_void,
    );

    pub fn mps_nn_optimizer_rmsprop_new(
        device_handle: *mut c_void,
        learning_rate: f32,
    ) -> *mut c_void;
    pub fn mps_nn_optimizer_rmsprop_new_with_options(
        device_handle: *mut c_void,
        decay: f64,
        epsilon: f32,
        optimizer_descriptor_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mps_nn_optimizer_rmsprop_decay(handle: *mut c_void) -> f64;
    pub fn mps_nn_optimizer_rmsprop_epsilon(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_rmsprop_encode_vector(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        input_gradient_vector_handle: *mut c_void,
        input_values_vector_handle: *mut c_void,
        input_sum_of_squares_vector_handle: *mut c_void,
        result_values_vector_handle: *mut c_void,
    );
    pub fn mps_nn_optimizer_rmsprop_encode_matrix(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        input_gradient_matrix_handle: *mut c_void,
        input_values_matrix_handle: *mut c_void,
        input_sum_of_squares_matrix_handle: *mut c_void,
        result_values_matrix_handle: *mut c_void,
    );

    pub fn mps_nn_optimizer_adam_new(device_handle: *mut c_void, learning_rate: f32)
        -> *mut c_void;
    pub fn mps_nn_optimizer_adam_new_with_options(
        device_handle: *mut c_void,
        beta1: f64,
        beta2: f64,
        epsilon: f32,
        time_step: usize,
        optimizer_descriptor_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mps_nn_optimizer_adam_beta1(handle: *mut c_void) -> f64;
    pub fn mps_nn_optimizer_adam_beta2(handle: *mut c_void) -> f64;
    pub fn mps_nn_optimizer_adam_epsilon(handle: *mut c_void) -> f32;
    pub fn mps_nn_optimizer_adam_time_step(handle: *mut c_void) -> usize;
    pub fn mps_nn_optimizer_adam_set_time_step(handle: *mut c_void, value: usize);
    pub fn mps_nn_optimizer_adam_encode_vector(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        input_gradient_vector_handle: *mut c_void,
        input_values_vector_handle: *mut c_void,
        input_momentum_vector_handle: *mut c_void,
        input_velocity_vector_handle: *mut c_void,
        result_values_vector_handle: *mut c_void,
    );
    pub fn mps_nn_optimizer_adam_encode_matrix(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        input_gradient_matrix_handle: *mut c_void,
        input_values_matrix_handle: *mut c_void,
        input_momentum_matrix_handle: *mut c_void,
        input_velocity_matrix_handle: *mut c_void,
        result_values_matrix_handle: *mut c_void,
    );
    pub fn mps_nn_optimizer_adam_encode_amsgrad_vector(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        input_gradient_vector_handle: *mut c_void,
        input_values_vector_handle: *mut c_void,
        input_momentum_vector_handle: *mut c_void,
        input_velocity_vector_handle: *mut c_void,
        maximum_velocity_vector_handle: *mut c_void,
        result_values_vector_handle: *mut c_void,
    );
    pub fn mps_nn_optimizer_adam_encode_amsgrad_matrix(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        input_gradient_matrix_handle: *mut c_void,
        input_values_matrix_handle: *mut c_void,
        input_momentum_matrix_handle: *mut c_void,
        input_velocity_matrix_handle: *mut c_void,
        maximum_velocity_matrix_handle: *mut c_void,
        result_values_matrix_handle: *mut c_void,
    );

    pub fn mps_rnn_descriptor_input_feature_channels(handle: *mut c_void) -> usize;
    pub fn mps_rnn_descriptor_set_input_feature_channels(handle: *mut c_void, value: usize);
    pub fn mps_rnn_descriptor_output_feature_channels(handle: *mut c_void) -> usize;
    pub fn mps_rnn_descriptor_set_output_feature_channels(handle: *mut c_void, value: usize);
    pub fn mps_rnn_descriptor_use_layer_input_unit_transform_mode(handle: *mut c_void) -> bool;
    pub fn mps_rnn_descriptor_set_use_layer_input_unit_transform_mode(
        handle: *mut c_void,
        value: bool,
    );
    pub fn mps_rnn_descriptor_use_float32_weights(handle: *mut c_void) -> bool;
    pub fn mps_rnn_descriptor_set_use_float32_weights(handle: *mut c_void, value: bool);
    pub fn mps_rnn_descriptor_layer_sequence_direction(handle: *mut c_void) -> usize;
    pub fn mps_rnn_descriptor_set_layer_sequence_direction(handle: *mut c_void, value: usize);

    pub fn mps_gru_descriptor_new(
        input_feature_channels: usize,
        output_feature_channels: usize,
    ) -> *mut c_void;
    pub fn mps_gru_descriptor_gate_pnorm_value(handle: *mut c_void) -> f32;
    pub fn mps_gru_descriptor_set_gate_pnorm_value(handle: *mut c_void, value: f32);
    pub fn mps_gru_descriptor_flip_output_gates(handle: *mut c_void) -> bool;
    pub fn mps_gru_descriptor_set_flip_output_gates(handle: *mut c_void, value: bool);

    pub fn mps_lstm_descriptor_new(
        input_feature_channels: usize,
        output_feature_channels: usize,
    ) -> *mut c_void;
    pub fn mps_lstm_descriptor_memory_weights_are_diagonal(handle: *mut c_void) -> bool;
    pub fn mps_lstm_descriptor_set_memory_weights_are_diagonal(handle: *mut c_void, value: bool);
    pub fn mps_lstm_descriptor_cell_to_output_neuron_type(handle: *mut c_void) -> usize;
    pub fn mps_lstm_descriptor_set_cell_to_output_neuron_type(handle: *mut c_void, value: usize);
    pub fn mps_lstm_descriptor_cell_to_output_neuron_param_a(handle: *mut c_void) -> f32;
    pub fn mps_lstm_descriptor_set_cell_to_output_neuron_param_a(handle: *mut c_void, value: f32);
    pub fn mps_lstm_descriptor_cell_to_output_neuron_param_b(handle: *mut c_void) -> f32;
    pub fn mps_lstm_descriptor_set_cell_to_output_neuron_param_b(handle: *mut c_void, value: f32);
    pub fn mps_lstm_descriptor_cell_to_output_neuron_param_c(handle: *mut c_void) -> f32;
    pub fn mps_lstm_descriptor_set_cell_to_output_neuron_param_c(handle: *mut c_void, value: f32);

    pub fn mps_rnn_recurrent_image_state_recurrent_output_image(
        handle: *mut c_void,
        layer_index: usize,
    ) -> *mut c_void;
    pub fn mps_rnn_recurrent_image_state_memory_cell_image(
        handle: *mut c_void,
        layer_index: usize,
    ) -> *mut c_void;

    pub fn mps_rnn_image_inference_layer_new(
        device_handle: *mut c_void,
        descriptor_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mps_rnn_image_inference_layer_new_stack(
        device_handle: *mut c_void,
        descriptor_count: usize,
        descriptor_handles: *const *mut c_void,
    ) -> *mut c_void;
    pub fn mps_rnn_image_inference_layer_input_feature_channels(handle: *mut c_void) -> usize;
    pub fn mps_rnn_image_inference_layer_output_feature_channels(handle: *mut c_void) -> usize;
    pub fn mps_rnn_image_inference_layer_number_of_layers(handle: *mut c_void) -> usize;
    pub fn mps_rnn_image_inference_layer_recurrent_output_is_temporary(handle: *mut c_void)
        -> bool;
    pub fn mps_rnn_image_inference_layer_set_recurrent_output_is_temporary(
        handle: *mut c_void,
        value: bool,
    );
    pub fn mps_rnn_image_inference_layer_store_all_intermediate_states(handle: *mut c_void)
        -> bool;
    pub fn mps_rnn_image_inference_layer_set_store_all_intermediate_states(
        handle: *mut c_void,
        value: bool,
    );
    pub fn mps_rnn_image_inference_layer_bidirectional_combine_mode(handle: *mut c_void) -> usize;
    pub fn mps_rnn_image_inference_layer_set_bidirectional_combine_mode(
        handle: *mut c_void,
        value: usize,
    );
    pub fn mps_rnn_image_inference_layer_encode_sequence(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        image_count: usize,
        source_image_handles: *const *mut c_void,
        destination_image_handles: *const *mut c_void,
        recurrent_input_state_handle: *mut c_void,
    ) -> *mut c_void;
}
