use apple_metal::{resource_options, MetalBuffer, MetalDevice};
use apple_mps::{
    acceleration_structure_status, cull_mode, data_type, feature_channel_format,
    intersection_data_type, intersection_type, polygon_type, ray_data_type, rnn_sequence_direction,
    CnnConvolutionDescriptor, CnnNeuronReluNode, CnnPoolingMaxNode, CnnSoftMaxNode,
    CnnUpsamplingNearestNode, Image, ImageDescriptor, NDArray, NDArrayDescriptor, NDArrayIdentity,
    NNGraph, NNImageNode, PolygonAccelerationStructure, RayIntersector, RnnSingleGateDescriptor,
    SVGF,
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PackedFloat3 {
    x: f32,
    y: f32,
    z: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PackedRayOriginDirection {
    origin: PackedFloat3,
    direction: PackedFloat3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct IntersectionDistancePrimitiveIndex {
    distance: f32,
    primitive_index: u32,
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

fn read_struct<T: Copy>(buffer: &MetalBuffer) -> T {
    let ptr = buffer.contents().expect("buffer contents").cast::<T>();
    unsafe { *ptr }
}

#[test]
fn ndarray_identity_smoke() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device.new_command_queue().expect("command queue");
    let descriptor =
        NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[2, 2]).expect("descriptor");
    let array = NDArray::new(&device, &descriptor).expect("ndarray");
    assert_eq!(array.number_of_dimensions(), 2);
    assert_eq!(array.length_of_dimension(0), 2);
    assert_eq!(array.length_of_dimension(1), 2);
    assert!(array.resource_size() > 0);
    if let Some(identity) = NDArrayIdentity::new(&device) {
        let destination_descriptor =
            NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[4])
                .expect("destination descriptor");
        let destination =
            NDArray::new(&device, &destination_descriptor).expect("destination ndarray");
        let command_buffer = queue.new_command_buffer().expect("command buffer");
        assert!(identity.reshape_into(Some(&command_buffer), &array, &[4], &destination));
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(destination.number_of_dimensions(), 1);
        assert_eq!(destination.length_of_dimension(0), 4);
    }
}

#[test]
fn ray_and_svgf_smoke() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device.new_command_queue().expect("command queue");

    let vertices = [
        [-1.0_f32, -1.0, 0.0, 0.0],
        [1.0_f32, -1.0, 0.0, 0.0],
        [0.0_f32, 1.0, 0.0, 0.0],
    ];
    let vertex_buffer = device
        .new_buffer(
            core::mem::size_of_val(&vertices),
            resource_options::STORAGE_MODE_SHARED,
        )
        .expect("vertex buffer");
    let _ = vertex_buffer.write_bytes(as_bytes(&vertices));

    let acceleration_structure =
        PolygonAccelerationStructure::new(&device).expect("polygon acceleration structure");
    acceleration_structure.set_polygon_type(polygon_type::TRIANGLE);
    acceleration_structure.set_vertex_stride(core::mem::size_of::<[f32; 4]>());
    acceleration_structure.set_index_type(data_type::UINT32);
    acceleration_structure.set_vertex_buffer(Some(&vertex_buffer));
    acceleration_structure.set_index_buffer(None);
    acceleration_structure.set_polygon_count(1);
    acceleration_structure.rebuild();
    assert_eq!(
        acceleration_structure.status(),
        acceleration_structure_status::BUILT
    );

    let ray = PackedRayOriginDirection {
        origin: PackedFloat3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        direction: PackedFloat3 {
            x: 0.0,
            y: 0.0,
            z: -1.0,
        },
    };
    let miss = IntersectionDistancePrimitiveIndex {
        distance: -1.0,
        primitive_index: u32::MAX,
    };

    let ray_buffer = device
        .new_buffer(
            core::mem::size_of::<PackedRayOriginDirection>(),
            resource_options::STORAGE_MODE_SHARED,
        )
        .expect("ray buffer");
    let intersection_buffer = device
        .new_buffer(
            core::mem::size_of::<IntersectionDistancePrimitiveIndex>(),
            resource_options::STORAGE_MODE_SHARED,
        )
        .expect("intersection buffer");
    let _ = ray_buffer.write_bytes(as_bytes(&[ray]));
    let _ = intersection_buffer.write_bytes(as_bytes(&[miss]));

    let intersector = RayIntersector::new(&device).expect("intersector");
    intersector.set_cull_mode(cull_mode::NONE);
    intersector.set_ray_data_type(ray_data_type::PACKED_ORIGIN_DIRECTION);
    intersector.set_intersection_data_type(intersection_data_type::DISTANCE_PRIMITIVE_INDEX);
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    intersector.encode_intersection(
        &command_buffer,
        intersection_type::NEAREST,
        &ray_buffer,
        0,
        &intersection_buffer,
        0,
        1,
        &acceleration_structure,
    );
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let intersection = read_struct::<IntersectionDistancePrimitiveIndex>(&intersection_buffer);
    assert!(
        (intersection.distance - 1.0).abs() < 1.0e-4,
        "{intersection:?}"
    );
    assert_eq!(intersection.primitive_index, 0);

    let svgf = SVGF::new(&device).expect("svgf");
    svgf.set_depth_weight(0.5);
    svgf.set_normal_weight(32.0);
    svgf.set_luminance_weight(1.5);
    svgf.set_channel_count(3);
    svgf.set_channel_count2(1);
    assert!((svgf.depth_weight() - 0.5).abs() < f32::EPSILON);
    assert!((svgf.normal_weight() - 32.0).abs() < f32::EPSILON);
    assert!((svgf.luminance_weight() - 1.5).abs() < f32::EPSILON);
    assert_eq!(svgf.channel_count(), 3);
    assert_eq!(svgf.channel_count2(), 1);
}

#[test]
fn neural_graph_and_descriptors_smoke() {
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
    assert!(pooling.result_image().is_some());
    let softmax = CnnSoftMaxNode::new(&relu_result).expect("softmax node");
    assert!(softmax.result_image().is_some());
    let upsampling = CnnUpsamplingNearestNode::new(&relu_result, 2, 2).expect("upsampling node");
    assert!(upsampling.result_image().is_some());

    let graph = NNGraph::new(&device, &relu_result, true).expect("graph");
    graph.set_format(feature_channel_format::FLOAT32);
    graph.use_default_destination_image_allocator();
    let _ = graph.source_image_count();

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
        assert!((actual - expected_value).abs() < 1.0e-4, "{output:?}");
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
        rnn_sequence_direction::BACKWARD
    );
}
