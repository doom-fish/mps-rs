use apple_metal::{resource_options, MetalBuffer, MetalDevice};
use apple_mps::{
    cnn_accumulator_precision_option, cnn_convolution_flags, nn_regularization_type,
    rnn_bidirectional_combine_mode, data_type, feature_channel_format,
    state_batch_increment_read_count, state_batch_resource_size, state_batch_synchronize,
    state_resource_type, CnnConvolution, CnnConvolutionDescriptor,
    CnnConvolutionWeightsAndBiasesState, GruDescriptor, Image, ImageDescriptor,
    LstmDescriptor, Matrix, MatrixDescriptor, NDArray, NDArrayDescriptor,
    NDArrayMatrixMultiplication, NNOptimizerAdam, NNOptimizerDescriptor,
    NNOptimizerRmsProp, NNOptimizerStochasticGradientDescent, RnnImageInferenceLayer,
    RnnSingleGateDescriptor, State, StateResourceList, Vector, VectorDescriptor,
};

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

fn buffer_with_f32_values_padded(device: &MetalDevice, values: &[f32], byte_len: usize) -> MetalBuffer {
    let buffer = device
        .new_buffer(
            byte_len.max(core::mem::size_of_val(values)),
            resource_options::STORAGE_MODE_SHARED,
        )
        .expect("buffer");
    let _ = buffer.write_bytes(as_bytes(values));
    buffer
}

fn buffer_with_f32_values(device: &MetalDevice, values: &[f32]) -> MetalBuffer {
    buffer_with_f32_values_padded(device, values, core::mem::size_of_val(values))
}

fn read_f32_values(buffer: &MetalBuffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents().expect("buffer contents").cast::<f32>();
    unsafe { core::slice::from_raw_parts(ptr, len).to_vec() }
}

fn vector_with_values(device: &MetalDevice, values: &[f32]) -> (MetalBuffer, Vector) {
    let buffer = buffer_with_f32_values(device, values);
    let descriptor = VectorDescriptor::contiguous(values.len(), data_type::FLOAT32).expect("vector desc");
    let vector = Vector::new_with_buffer(&buffer, descriptor).expect("vector");
    (buffer, vector)
}

fn matrix_with_values(device: &MetalDevice, rows: usize, columns: usize, values: &[f32]) -> (MetalBuffer, Matrix) {
    let buffer = buffer_with_f32_values(device, values);
    let descriptor = MatrixDescriptor::contiguous(rows, columns, data_type::FLOAT32).expect("matrix desc");
    let matrix = Matrix::new_with_buffer(&buffer, descriptor).expect("matrix");
    (buffer, matrix)
}

fn approx_eq(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (actual_value, expected_value) in actual.iter().zip(expected) {
        assert!((actual_value - expected_value).abs() < 1.0e-3, "actual={actual:?} expected={expected:?}");
    }
}

#[test]
fn ndarray_matrix_multiplication_smoke() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device.new_command_queue().expect("command queue");

    let descriptor = NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[2, 2, 1, 1]).expect("descriptor");
    let template = NDArray::new(&device, &descriptor).expect("template ndarray");
    let byte_len = template.resource_size();
    let rows = descriptor.length_of_dimension(1);
    let row_stride_floats = byte_len / core::mem::size_of::<f32>() / rows;
    let left_buffer = buffer_with_f32_values_padded(
        &device,
        &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0],
        byte_len,
    );
    let right_buffer = buffer_with_f32_values_padded(
        &device,
        &[5.0, 6.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0],
        byte_len,
    );
    let destination_buffer = buffer_with_f32_values_padded(&device, &[0.0; 8], byte_len);
    let left = NDArray::new_with_buffer(&left_buffer, 0, &descriptor).expect("left ndarray");
    let right = NDArray::new_with_buffer(&right_buffer, 0, &descriptor).expect("right ndarray");
    let destination = NDArray::new_with_buffer(&destination_buffer, 0, &descriptor).expect("destination ndarray");

    let kernel = NDArrayMatrixMultiplication::new(&device, 2).expect("ndarray matmul");
    kernel.set_alpha(1.0);
    kernel.set_beta(0.0);
    assert!((kernel.alpha() - 1.0).abs() < f64::EPSILON);
    assert!((kernel.beta() - 0.0).abs() < f64::EPSILON);

    let command_buffer = queue.new_command_buffer().expect("command buffer");
    kernel.encode_to_destination(&command_buffer, &[&left, &right], &destination);
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let padded_output = read_f32_values(&destination_buffer, row_stride_floats * rows);
    let output = vec![
        padded_output[0],
        padded_output[1],
        padded_output[row_stride_floats],
        padded_output[row_stride_floats + 1],
    ];
    approx_eq(&output, &[19.0, 22.0, 43.0, 50.0]);
}

#[test]
#[allow(clippy::too_many_lines)]
fn state_and_optimizer_smoke() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device.new_command_queue().expect("command queue");

    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let temporary_a = State::temporary_with_buffer_size(&command_buffer, 32).expect("temporary state a");
    let temporary_b = State::temporary_with_buffer_size(&command_buffer, 64).expect("temporary state b");
    assert!(temporary_a.is_temporary());
    assert_eq!(temporary_a.buffer_size_at_index(0), 32);
    assert_eq!(temporary_a.resource_type_at_index(0), state_resource_type::BUFFER);
    let unique_count = state_batch_increment_read_count(&[&temporary_a, &temporary_a, &temporary_b], 2);
    assert_eq!(unique_count, 2);
    assert_eq!(temporary_a.read_count(), 3);
    assert_eq!(temporary_b.read_count(), 3);
    let _ = state_batch_resource_size(&[&temporary_a, &temporary_b]);
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let resource_list = StateResourceList::new().expect("resource list");
    resource_list.append_buffer(16);
    resource_list.append_buffer(8);
    let persistent_state = State::new_with_resource_list(&device, &resource_list).expect("persistent state");
    assert_eq!(persistent_state.resource_count(), 2);
    assert_eq!(persistent_state.buffer_size_at_index(0), 16);
    assert_eq!(persistent_state.buffer_size_at_index(1), 8);
    assert_eq!(persistent_state.resource_type_at_index(1), state_resource_type::BUFFER);
    let _ = persistent_state.resource_size();
    let sync_command_buffer = queue.new_command_buffer().expect("sync command buffer");
    state_batch_synchronize(&[&persistent_state], &sync_command_buffer);
    sync_command_buffer.commit();
    sync_command_buffer.wait_until_completed();

    let descriptor = NNOptimizerDescriptor::with_gradient_clipping(
        0.25,
        1.0,
        true,
        1.5,
        -1.5,
        nn_regularization_type::NONE,
        0.0,
    )
    .expect("optimizer descriptor");
    assert_eq!(descriptor.regularization_type(), nn_regularization_type::NONE);
    assert!(descriptor.apply_gradient_clipping());
    descriptor.set_learning_rate(0.5);
    assert!((descriptor.learning_rate() - 0.5).abs() < f32::EPSILON);
    descriptor.set_gradient_rescale(0.75);
    assert!((descriptor.gradient_rescale() - 0.75).abs() < f32::EPSILON);
    descriptor.set_gradient_clip_max(2.0);
    descriptor.set_gradient_clip_min(-2.0);
    descriptor.set_regularization_scale(0.125);
    descriptor.set_regularization_type(nn_regularization_type::L2);
    assert!((descriptor.regularization_scale() - 0.125).abs() < f32::EPSILON);
    assert_eq!(descriptor.regularization_type(), nn_regularization_type::L2);

    let gradient_values = [0.1_f32, -0.2];
    let source_values = [1.0_f32, -1.0];

    let (gradient_buffer, gradient_vector) = vector_with_values(&device, &gradient_values);
    let (values_buffer, values_vector) = vector_with_values(&device, &source_values);
    let (sgd_result_buffer, sgd_result_vector) = vector_with_values(&device, &[0.0, 0.0]);
    let sgd = NNOptimizerStochasticGradientDescent::new(&device, 0.5).expect("sgd");
    let sgd_base = sgd.as_optimizer().expect("sgd base");
    assert!((sgd_base.learning_rate() - 0.5).abs() < f32::EPSILON);
    let sgd_command_buffer = queue.new_command_buffer().expect("sgd command buffer");
    sgd.encode_vector(
        &sgd_command_buffer,
        &gradient_vector,
        &values_vector,
        None,
        &sgd_result_vector,
    );
    sgd_command_buffer.commit();
    sgd_command_buffer.wait_until_completed();
    let sgd_output = read_f32_values(&sgd_result_buffer, 2);
    approx_eq(&sgd_output, &[0.95, -0.9]);

    let (_gradient_matrix_buffer, gradient_matrix) = matrix_with_values(&device, 1, 2, &gradient_values);
    let (_values_matrix_buffer, values_matrix) = matrix_with_values(&device, 1, 2, &source_values);
    let (sgd_matrix_result_buffer, sgd_matrix_result) = matrix_with_values(&device, 1, 2, &[0.0, 0.0]);
    let sgd_matrix_command_buffer = queue.new_command_buffer().expect("sgd matrix command buffer");
    sgd.encode_matrix(
        &sgd_matrix_command_buffer,
        &gradient_matrix,
        &values_matrix,
        None,
        &sgd_matrix_result,
    );
    sgd_matrix_command_buffer.commit();
    sgd_matrix_command_buffer.wait_until_completed();
    let sgd_matrix_output = read_f32_values(&sgd_matrix_result_buffer, 2);
    approx_eq(&sgd_matrix_output, &[0.95, -0.9]);

    let _ = gradient_buffer;
    let _ = values_buffer;

    let rms_descriptor = NNOptimizerDescriptor::new(0.25, 1.0, nn_regularization_type::NONE, 0.0)
        .expect("rms descriptor");
    let rms = NNOptimizerRmsProp::new_with_options(&device, 0.9, 1.0e-8, &rms_descriptor).expect("rmsprop");
    assert!((rms.decay() - 0.9).abs() < f64::EPSILON);
    let (_sumsq_buffer, sumsq_vector) = vector_with_values(&device, &[0.0, 0.0]);
    let (rms_result_buffer, rms_result_vector) = vector_with_values(&device, &[0.0, 0.0]);
    let rms_command_buffer = queue.new_command_buffer().expect("rms command buffer");
    rms.encode_vector(
        &rms_command_buffer,
        &gradient_vector,
        &values_vector,
        &sumsq_vector,
        &rms_result_vector,
    );
    rms_command_buffer.commit();
    rms_command_buffer.wait_until_completed();
    let rms_output = read_f32_values(&rms_result_buffer, 2);
    let rms_expected: Vec<f32> = source_values
        .iter()
        .zip(gradient_values)
        .map(|(value, gradient)| {
            let sumsq = 0.1 * gradient * gradient;
            value - 0.25 * gradient / (sumsq.sqrt() + 1.0e-8)
        })
        .collect();
    approx_eq(&rms_output, &rms_expected);

    let adam_descriptor = NNOptimizerDescriptor::new(0.25, 1.0, nn_regularization_type::NONE, 0.0)
        .expect("adam descriptor");
    let adam = NNOptimizerAdam::new_with_options(&device, 0.9, 0.999, 1.0e-8, 0, &adam_descriptor)
        .expect("adam");
    assert!((adam.beta1() - 0.9).abs() < f64::EPSILON);
    assert!((adam.beta2() - 0.999).abs() < f64::EPSILON);
    let (_momentum_buffer, momentum_vector) = vector_with_values(&device, &[0.0, 0.0]);
    let (_velocity_buffer, velocity_vector) = vector_with_values(&device, &[0.0, 0.0]);
    let (adam_result_buffer, adam_result_vector) = vector_with_values(&device, &[0.0, 0.0]);
    let adam_command_buffer = queue.new_command_buffer().expect("adam command buffer");
    adam.encode_vector(
        &adam_command_buffer,
        &gradient_vector,
        &values_vector,
        &momentum_vector,
        &velocity_vector,
        &adam_result_vector,
    );
    adam_command_buffer.commit();
    adam_command_buffer.wait_until_completed();
    let adam_output = read_f32_values(&adam_result_buffer, 2);
    approx_eq(&adam_output, &[0.75, -0.75]);
}

#[test]
fn convolution_and_rnn_smoke() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device.new_command_queue().expect("command queue");

    let convolution_descriptor = CnnConvolutionDescriptor::new(1, 1, 1, 1).expect("conv descriptor");
    assert_eq!(convolution_descriptor.input_feature_channels(), 1);
    assert_eq!(convolution_descriptor.output_feature_channels(), 1);
    let convolution = CnnConvolution::new(
        &device,
        &convolution_descriptor,
        &[2.0],
        Some(&[0.5]),
        cnn_convolution_flags::NONE,
    )
    .expect("convolution");
    convolution.set_accumulator_precision_option(cnn_accumulator_precision_option::FLOAT);
    assert_eq!(convolution.accumulator_precision_option(), cnn_accumulator_precision_option::FLOAT);
    assert_eq!(convolution.input_feature_channels(), 1);
    assert_eq!(convolution.output_feature_channels(), 1);

    let weights_buffer = buffer_with_f32_values(&device, &[2.0]);
    let biases_buffer = buffer_with_f32_values(&device, &[0.5]);
    let offsets_weights_buffer = buffer_with_f32_values(&device, &[0.0, 2.0]);
    let offsets_biases_buffer = buffer_with_f32_values(&device, &[0.0, 0.5]);
    let state_with_buffers = CnnConvolutionWeightsAndBiasesState::new_with_buffers(
        &weights_buffer,
        Some(&biases_buffer),
    )
    .expect("state with buffers");
    assert_eq!(state_with_buffers.weights_offset(), 0);
    assert_eq!(state_with_buffers.biases_offset(), 0);
    let offset_state = CnnConvolutionWeightsAndBiasesState::new_with_offsets(
        &offsets_weights_buffer,
        4,
        Some(&offsets_biases_buffer),
        4,
        &convolution_descriptor,
    )
    .expect("state with offsets");
    assert_eq!(offset_state.weights_offset(), 4);
    assert_eq!(offset_state.biases_offset(), 4);
    let allocated_state = CnnConvolutionWeightsAndBiasesState::new_with_device(&device, &convolution_descriptor)
        .expect("allocated state");
    assert_eq!(allocated_state.weights_offset(), 0);
    assert_eq!(allocated_state.biases_offset(), 0);

    let image_descriptor = ImageDescriptor::new(2, 2, 1, feature_channel_format::FLOAT32);
    let source = Image::new(&device, image_descriptor).expect("source image");
    let destination = Image::new(&device, image_descriptor).expect("destination image");
    source.write_f32(&[1.0, 2.0, 3.0, 4.0]).expect("write source");

    let convolution_command_buffer = queue.new_command_buffer().expect("conv command buffer");
    convolution.encode_image(&convolution_command_buffer, &source, &destination);
    convolution_command_buffer.commit();
    convolution_command_buffer.wait_until_completed();
    let convolution_output = destination.read_f32().expect("conv output");
    approx_eq(&convolution_output, &[2.5, 4.5, 6.5, 8.5]);

    let gru = GruDescriptor::new(1, 1).expect("gru descriptor");
    gru.set_gate_pnorm_value(2.0);
    gru.set_flip_output_gates(true);
    assert!((gru.gate_pnorm_value() - 2.0).abs() < f32::EPSILON);
    assert!(gru.flip_output_gates());

    let lstm = LstmDescriptor::new(1, 1).expect("lstm descriptor");
    lstm.set_memory_weights_are_diagonal(true);
    lstm.set_cell_to_output_neuron_param_a(0.5);
    lstm.set_cell_to_output_neuron_param_b(1.5);
    lstm.set_cell_to_output_neuron_param_c(2.5);
    assert!(lstm.memory_weights_are_diagonal());
    assert!((lstm.cell_to_output_neuron_param_a() - 0.5).abs() < f32::EPSILON);
    assert!((lstm.cell_to_output_neuron_param_b() - 1.5).abs() < f32::EPSILON);
    assert!((lstm.cell_to_output_neuron_param_c() - 2.5).abs() < f32::EPSILON);

    let single_gate = RnnSingleGateDescriptor::new(1, 1).expect("single gate descriptor");
    single_gate.set_use_layer_input_unit_transform_mode(true);
    let base_descriptor = single_gate.as_descriptor().expect("base descriptor");
    assert_eq!(base_descriptor.input_feature_channels(), 1);
    let layer = RnnImageInferenceLayer::new(&device, &base_descriptor).expect("rnn layer");
    let stacked_layer = RnnImageInferenceLayer::new_stack(&device, &[&base_descriptor]).expect("stacked rnn layer");
    assert_eq!(stacked_layer.number_of_layers(), 1);
    layer.set_recurrent_output_is_temporary(false);
    layer.set_store_all_intermediate_states(true);
    layer.set_bidirectional_combine_mode(rnn_bidirectional_combine_mode::ADD);
    assert_eq!(layer.bidirectional_combine_mode(), rnn_bidirectional_combine_mode::ADD);

    let seq_descriptor = ImageDescriptor::new(1, 1, 1, feature_channel_format::FLOAT32);
    let src0 = Image::new(&device, seq_descriptor).expect("src0");
    let src1 = Image::new(&device, seq_descriptor).expect("src1");
    let dst0 = Image::new(&device, seq_descriptor).expect("dst0");
    let dst1 = Image::new(&device, seq_descriptor).expect("dst1");
    src0.write_f32(&[0.25]).expect("write src0");
    src1.write_f32(&[0.75]).expect("write src1");

    let rnn_command_buffer = queue.new_command_buffer().expect("rnn command buffer");
    let recurrent_state = layer
        .encode_sequence(&rnn_command_buffer, &[&src0, &src1], &[&dst0, &dst1], None)
        .expect("recurrent state");
    rnn_command_buffer.commit();
    rnn_command_buffer.wait_until_completed();

    approx_eq(&dst0.read_f32().expect("dst0 output"), &[0.25]);
    approx_eq(&dst1.read_f32().expect("dst1 output"), &[0.75]);
    let recurrent_output = recurrent_state
        .recurrent_output_image_for_layer_index(0)
        .expect("recurrent output image");
    assert_eq!(recurrent_output.width(), 1);
    assert_eq!(recurrent_output.height(), 1);
    assert_eq!(recurrent_output.feature_channels(), 1);
}
