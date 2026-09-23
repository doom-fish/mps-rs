use apple_metal::{resource_options, MetalBuffer, MetalDevice};
use apple_mps::{
    cnn_convolution_flags, data_type, feature_channel_format, CnnConvolution,
    CnnConvolutionDescriptor, CnnConvolutionWeightsAndBiasesState, CnnNeuronReluNode, Error, Image,
    ImageDescriptor, Matrix, MatrixDescriptor, NNGraph, NNImageNode,
    NNOptimizerStochasticGradientDescent, Vector, VectorDescriptor,
};

fn device() -> MetalDevice {
    MetalDevice::system_default().expect("no Metal device available")
}

fn buffer(device: &MetalDevice, length: usize) -> MetalBuffer {
    device
        .new_buffer(length, resource_options::STORAGE_MODE_SHARED)
        .expect("buffer")
}

fn descriptor(input: usize, output: usize, groups: usize) -> CnnConvolutionDescriptor {
    let descriptor = CnnConvolutionDescriptor::new(1, 1, input, output).expect("descriptor");
    descriptor.set_groups(groups).expect("valid groups");
    descriptor
}

fn set_groups(input: usize, output: usize, groups: usize) -> apple_mps::Result<()> {
    let descriptor = CnnConvolutionDescriptor::new(1, 1, input, output).expect("descriptor");
    let result = descriptor.set_groups(groups);
    if result.is_err() {
        assert_eq!(descriptor.groups(), 1);
    }
    result
}

fn convolution(
    device: &MetalDevice,
    descriptor: &CnnConvolutionDescriptor,
    weights: usize,
) -> apple_mps::Result<CnnConvolution> {
    CnnConvolution::new(
        device,
        descriptor,
        &vec![1.0; weights],
        None,
        cnn_convolution_flags::NONE,
    )
}

fn vector(device: &MetalDevice, length: usize, data: u32) -> (MetalBuffer, Vector) {
    let descriptor = VectorDescriptor::contiguous(length, data).expect("descriptor");
    let buffer = buffer(device, descriptor.vector_bytes.max(4));
    let vector = Vector::new_with_buffer(&buffer, descriptor).expect("vector");
    (buffer, vector)
}

fn matrix(device: &MetalDevice, rows: usize, columns: usize) -> (MetalBuffer, Matrix) {
    let descriptor =
        MatrixDescriptor::contiguous(rows, columns, data_type::FLOAT32).expect("descriptor");
    let buffer = buffer(device, descriptor.matrix_bytes.max(4));
    let matrix = Matrix::new_with_buffer(&buffer, descriptor).expect("matrix");
    (buffer, matrix)
}

fn image(device: &MetalDevice, channels: usize) -> Image {
    Image::new(
        device,
        ImageDescriptor::new(2, 2, channels, feature_channel_format::FLOAT32),
    )
    .expect("image")
}

#[test]
fn convolution_groups_must_divide_the_channels() {
    let device = device();
    assert_eq!(
        set_groups(4, 4, 0),
        Err(Error::InvalidArgument("groups must be at least 1"))
    );
    assert_eq!(
        set_groups(4, 4, 8),
        Err(Error::InvalidArgument(
            "groups exceeds the input feature channels"
        ))
    );
    let not_divisible =
        Error::InvalidArgument("input and output feature channels must be divisible by groups");
    assert_eq!(set_groups(8, 8, 3), Err(not_divisible.clone()));
    assert_eq!(set_groups(8, 7, 2), Err(not_divisible));
    let per_group = Error::InvalidArgument(
        "with several groups, each group needs a multiple of 4 input and output channels",
    );
    assert_eq!(set_groups(4, 4, 2), Err(per_group.clone()));
    assert_eq!(set_groups(8, 6, 2), Err(per_group));
    assert_eq!(
        convolution(&device, &descriptor(8, 8, 2), 31).err(),
        Some(Error::DimensionMismatch {
            field: "kernel_weights",
            expected: 32,
            actual: 31,
        })
    );
    let grouped = convolution(&device, &descriptor(8, 8, 2), 32).expect("two groups of four");
    assert_eq!(grouped.groups(), 2);
    let odd = convolution(&device, &descriptor(3, 5, 1), 15).expect("one group of any size");
    assert_eq!(odd.input_feature_channels(), 3);
}

#[test]
fn convolution_descriptors_and_arguments_are_checked() {
    let device = device();
    let zero_kernel = CnnConvolutionDescriptor::new(0, 1, 1, 1).expect("descriptor");
    assert!(matches!(
        convolution(&device, &zero_kernel, 1),
        Err(Error::InvalidArgument(_))
    ));
    let no_input = CnnConvolutionDescriptor::new(1, 1, 0, 1).expect("descriptor");
    assert!(matches!(
        convolution(&device, &no_input, 1),
        Err(Error::InvalidArgument(_))
    ));
    let strided = descriptor(1, 1, 1);
    strided.set_stride_in_pixels_x(0);
    assert_eq!(
        convolution(&device, &strided, 1).err(),
        Some(Error::InvalidArgument("strides must be at least 1"))
    );
    let valid = descriptor(1, 2, 1);
    assert!(matches!(
        CnnConvolution::new(&device, &valid, &[1.0, 1.0], None, 1),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(
        CnnConvolution::new(
            &device,
            &valid,
            &[1.0, 1.0],
            Some(&[0.5]),
            cnn_convolution_flags::NONE
        )
        .err(),
        Some(Error::DimensionMismatch {
            field: "bias_terms",
            expected: 2,
            actual: 1,
        })
    );
}

#[test]
fn weights_and_biases_buffers_are_checked() {
    let device = device();
    let conv = CnnConvolutionDescriptor::new(3, 3, 4, 4).expect("descriptor");
    let weights = buffer(&device, 576);
    let biases = buffer(&device, 16);
    assert_eq!(
        CnnConvolutionWeightsAndBiasesState::new_with_offsets(
            &buffer(&device, 4),
            0,
            None,
            0,
            &conv
        )
        .err(),
        Some(Error::BufferTooSmall {
            required: 576,
            length: 4,
        })
    );
    assert_eq!(
        CnnConvolutionWeightsAndBiasesState::new_with_offsets(&weights, 4, None, 0, &conv).err(),
        Some(Error::BufferTooSmall {
            required: 580,
            length: 576,
        })
    );
    assert_eq!(
        CnnConvolutionWeightsAndBiasesState::new_with_offsets(&weights, 2, None, 0, &conv).err(),
        Some(Error::Misaligned {
            field: "weights_offset",
            value: 2,
            alignment: 4,
        })
    );
    assert_eq!(
        CnnConvolutionWeightsAndBiasesState::new_with_offsets(&weights, 0, Some(&biases), 4, &conv)
            .err(),
        Some(Error::BufferTooSmall {
            required: 20,
            length: 16,
        })
    );
    let state =
        CnnConvolutionWeightsAndBiasesState::new_with_offsets(&weights, 0, Some(&biases), 0, &conv)
            .expect("exact fit");
    assert_eq!(state.weights_offset(), 0);
    assert!(conv.set_groups(0).is_err());
    let allocated =
        CnnConvolutionWeightsAndBiasesState::new_with_device(&device, &conv).expect("state");
    assert_eq!(allocated.biases_offset(), 0);
}

#[test]
fn convolution_encode_checks_image_channels() {
    let device = device();
    let queue = device.new_command_queue().expect("queue");
    let four = convolution(&device, &descriptor(4, 4, 1), 16).expect("convolution");
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    assert_eq!(
        four.encode_image(&command_buffer, &image(&device, 1), &image(&device, 4)),
        Err(Error::DimensionMismatch {
            field: "source feature channels",
            expected: 4,
            actual: 1,
        })
    );
    assert_eq!(
        four.encode_image(&command_buffer, &image(&device, 4), &image(&device, 1)),
        Err(Error::DimensionMismatch {
            field: "destination feature channels",
            expected: 4,
            actual: 1,
        })
    );
    let unwritten = image(&device, 1);
    let destination = image(&device, 1);
    let one = CnnConvolution::new(
        &device,
        &descriptor(1, 1, 1),
        &[2.0],
        Some(&[0.5]),
        cnn_convolution_flags::NONE,
    )
    .expect("convolution");
    one.encode_image(&command_buffer, &unwritten, &destination)
        .expect("an unwritten source no longer aborts");
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
    assert_eq!(destination.read_f32().expect("output").len(), 4);
}

#[test]
fn optimizer_inputs_must_share_shape_and_float32() {
    let device = device();
    let queue = device.new_command_queue().expect("queue");
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let sgd = NNOptimizerStochasticGradientDescent::new(&device, 0.5).expect("sgd");
    let (_a, four) = vector(&device, 4, data_type::FLOAT32);
    let (_b, two) = vector(&device, 2, data_type::FLOAT32);
    let (_c, other) = vector(&device, 4, data_type::FLOAT32);
    assert_eq!(
        sgd.encode_vector(&command_buffer, &four, &two, None, &other),
        Err(Error::DimensionMismatch {
            field: "vector length",
            expected: 4,
            actual: 2,
        })
    );
    assert_eq!(
        sgd.encode_vector(&command_buffer, &four, &other, Some(&two), &other),
        Err(Error::DimensionMismatch {
            field: "vector length",
            expected: 4,
            actual: 2,
        })
    );
    let (_d, int) = vector(&device, 4, data_type::INT32);
    assert_eq!(
        sgd.encode_vector(&command_buffer, &int, &int, None, &int),
        Err(Error::UnsupportedDataType(data_type::INT32))
    );
    let (_e, square) = matrix(&device, 2, 2);
    let (_f, tall) = matrix(&device, 4, 1);
    assert_eq!(
        sgd.encode_matrix(&command_buffer, &square, &tall, None, &square),
        Err(Error::DimensionMismatch {
            field: "matrix rows",
            expected: 2,
            actual: 4,
        })
    );
    sgd.encode_vector(&command_buffer, &four, &other, None, &other)
        .expect("matching vectors");
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
}

#[test]
fn graph_encode_needs_every_source_image() {
    let device = device();
    let queue = device.new_command_queue().expect("queue");
    let input = NNImageNode::new().expect("input");
    let relu = CnnNeuronReluNode::new(&input, 0.0).expect("relu");
    let graph = NNGraph::new(&device, &relu.result_image().expect("result"), true).expect("graph");
    assert_eq!(graph.source_image_count(), 1);
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    assert!(graph.encode(&command_buffer, &[]).is_none());
    let source = image(&device, 1);
    assert!(graph.encode(&command_buffer, &[&source, &source]).is_none());
    let result = graph
        .encode(&command_buffer, &[&source])
        .expect("one source");
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
    assert_eq!(result.width(), 2);
}
