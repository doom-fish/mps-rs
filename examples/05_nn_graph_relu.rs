use apple_metal::MetalDevice;
use apple_mps::{
    feature_channel_format, rnn_sequence_direction, CnnConvolutionDescriptor, CnnNeuronReluNode,
    CnnPoolingMaxNode, CnnSoftMaxNode, CnnUpsamplingNearestNode, Image, ImageDescriptor, NNGraph,
    NNImageNode, RnnSingleGateDescriptor,
};

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device.new_command_queue().expect("command queue");

    let input_node = NNImageNode::new().expect("input node");
    input_node.set_format(feature_channel_format::FLOAT32);
    let relu = CnnNeuronReluNode::new(&input_node, 0.0).expect("relu node");
    let relu_result = relu.result_image().expect("relu result image");
    relu_result.set_format(feature_channel_format::FLOAT32);
    relu_result.set_synchronize_resource(true);
    relu_result.use_default_allocator();
    let pooling = CnnPoolingMaxNode::new(&relu_result, 2, 2).expect("pooling node");
    assert!(
        pooling.result_image().is_some(),
        "pooling result image should exist"
    );
    let softmax = CnnSoftMaxNode::new(&relu_result).expect("softmax node");
    assert!(
        softmax.result_image().is_some(),
        "softmax result image should exist"
    );
    let upsampling = CnnUpsamplingNearestNode::new(&relu_result, 2, 2).expect("upsampling node");
    assert!(
        upsampling.result_image().is_some(),
        "upsampling result image should exist"
    );

    let graph = NNGraph::new(&device, &relu_result, true).expect("graph");
    graph.set_format(feature_channel_format::FLOAT32);
    graph.use_default_destination_image_allocator();

    let descriptor = ImageDescriptor::new(2, 2, 1, feature_channel_format::FLOAT32);
    let source = Image::new(&device, descriptor).expect("source image");
    source
        .write_f32(&[-1.0, 0.5, 2.0, -0.25])
        .expect("write source image");

    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let result = graph
        .encode(&command_buffer, &[&source])
        .expect("graph encode");
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let output = result.read_f32().expect("read graph output");
    let expected = [0.0_f32, 0.5, 2.0, 0.0];
    for (actual, expected_value) in output.iter().zip(expected) {
        assert!(
            (actual - expected_value).abs() < 1.0e-4,
            "unexpected relu graph output: {output:?}"
        );
    }

    let convolution = CnnConvolutionDescriptor::new(3, 3, 1, 4).expect("convolution descriptor");
    convolution.set_stride_in_pixels_x(2);
    convolution.set_stride_in_pixels_y(1);
    convolution.set_groups(1);
    convolution.set_dilation_rate_x(1);
    convolution.set_dilation_rate_y(2);
    assert_eq!(convolution.kernel_width(), 3);
    assert_eq!(convolution.kernel_height(), 3);
    assert_eq!(convolution.stride_in_pixels_x(), 2);
    assert_eq!(convolution.stride_in_pixels_y(), 1);
    assert_eq!(convolution.groups(), 1);
    assert_eq!(convolution.dilation_rate_x(), 1);
    assert_eq!(convolution.dilation_rate_y(), 2);

    let rnn = RnnSingleGateDescriptor::new(3, 5).expect("rnn descriptor");
    rnn.set_use_layer_input_unit_transform_mode(true);
    rnn.set_use_float32_weights(true);
    rnn.set_layer_sequence_direction(rnn_sequence_direction::BACKWARD);
    assert_eq!(rnn.input_feature_channels(), 3);
    assert_eq!(rnn.output_feature_channels(), 5);
    assert!(rnn.use_layer_input_unit_transform_mode());
    assert!(rnn.use_float32_weights());
    assert_eq!(
        rnn.layer_sequence_direction(),
        rnn_sequence_direction::BACKWARD,
        "expected backward RNN sequence direction"
    );

    println!(
        "nn smoke passed: relu={output:?} source_images={}",
        graph.source_image_count()
    );
}
