import Metal
import MetalPerformanceShaders

private func mps_nn_source_images(
    count: Int,
    handles: UnsafePointer<UnsafeMutableRawPointer?>
) -> [MPSImage]? {
    guard count >= 0 else { return nil }
    var images = [MPSImage]()
    images.reserveCapacity(count)
    for index in 0..<count {
        guard let image: MPSImage = mps_borrow(handles[index]) else {
            return nil
        }
        images.append(image)
    }
    return images
}

@_cdecl("mps_nn_image_node_new")
public func mps_nn_image_node_new() -> UnsafeMutableRawPointer? {
    mps_retain(MPSNNImageNode(handle: nil))
}

@_cdecl("mps_nn_image_node_exported")
public func mps_nn_image_node_exported() -> UnsafeMutableRawPointer? {
    mps_retain(MPSNNImageNode.exportedNode(with: nil))
}

@_cdecl("mps_nn_image_node_format")
public func mps_nn_image_node_format(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let imageNode: MPSNNImageNode = mps_borrow(handle) else { return 0 }
    return imageNode.format.rawValue
}

@_cdecl("mps_nn_image_node_set_format")
public func mps_nn_image_node_set_format(
    _ handle: UnsafeMutableRawPointer?,
    _ formatRaw: UInt
) {
    guard let imageNode: MPSNNImageNode = mps_borrow(handle),
          let format = mps_channel_format(formatRaw)
    else {
        return
    }
    imageNode.format = format
}

@_cdecl("mps_nn_image_node_export_from_graph")
public func mps_nn_image_node_export_from_graph(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let imageNode: MPSNNImageNode = mps_borrow(handle) else { return false }
    return imageNode.exportFromGraph
}

@_cdecl("mps_nn_image_node_set_export_from_graph")
public func mps_nn_image_node_set_export_from_graph(
    _ handle: UnsafeMutableRawPointer?,
    _ export: Bool
) {
    guard let imageNode: MPSNNImageNode = mps_borrow(handle) else { return }
    imageNode.exportFromGraph = export
}

@_cdecl("mps_nn_image_node_synchronize_resource")
public func mps_nn_image_node_synchronize_resource(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let imageNode: MPSNNImageNode = mps_borrow(handle) else { return false }
    return imageNode.synchronizeResource
}

@_cdecl("mps_nn_image_node_set_synchronize_resource")
public func mps_nn_image_node_set_synchronize_resource(
    _ handle: UnsafeMutableRawPointer?,
    _ synchronize: Bool
) {
    guard let imageNode: MPSNNImageNode = mps_borrow(handle) else { return }
    imageNode.synchronizeResource = synchronize
}

@_cdecl("mps_nn_image_node_use_default_allocator")
public func mps_nn_image_node_use_default_allocator(_ handle: UnsafeMutableRawPointer?) {
    guard let imageNode: MPSNNImageNode = mps_borrow(handle) else { return }
    imageNode.imageAllocator = MPSImage.defaultAllocator()
}

@_cdecl("mps_cnn_neuron_relu_node_new")
public func mps_cnn_neuron_relu_node_new(
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ a: Float
) -> UnsafeMutableRawPointer? {
    guard let sourceNode: MPSNNImageNode = mps_borrow(sourceHandle) else { return nil }
    return mps_retain(MPSCNNNeuronReLUNode(source: sourceNode, a: a))
}

@_cdecl("mps_cnn_pooling_max_node_new")
public func mps_cnn_pooling_max_node_new(
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ filterSize: Int,
    _ stride: Int
) -> UnsafeMutableRawPointer? {
    guard let sourceNode: MPSNNImageNode = mps_borrow(sourceHandle) else { return nil }
    return mps_retain(MPSCNNPoolingMaxNode(source: sourceNode, filterSize: filterSize, stride: stride))
}

@_cdecl("mps_cnn_softmax_node_new")
public func mps_cnn_softmax_node_new(
    _ sourceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let sourceNode: MPSNNImageNode = mps_borrow(sourceHandle) else { return nil }
    return mps_retain(MPSCNNSoftMaxNode(source: sourceNode))
}

@_cdecl("mps_cnn_upsampling_nearest_node_new")
public func mps_cnn_upsampling_nearest_node_new(
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ scaleX: Int,
    _ scaleY: Int
) -> UnsafeMutableRawPointer? {
    guard let sourceNode: MPSNNImageNode = mps_borrow(sourceHandle) else { return nil }
    return mps_retain(
        MPSCNNUpsamplingNearestNode(
            source: sourceNode,
            integerScaleFactorX: scaleX,
            integerScaleFactorY: scaleY
        )
    )
}

@_cdecl("mps_nn_filter_node_result_image")
public func mps_nn_filter_node_result_image(
    _ handle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let filterNode: MPSNNFilterNode = mps_borrow(handle) else { return nil }
    return mps_retain(filterNode.resultImage)
}

@_cdecl("mps_nn_graph_new")
public func mps_nn_graph_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ resultImageHandle: UnsafeMutableRawPointer?,
    _ resultImageIsNeeded: Bool
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let resultImage: MPSNNImageNode = mps_borrow(resultImageHandle)
    else {
        return nil
    }
    guard let graph = MPSNNGraph(
        device: device,
        resultImage: resultImage,
        resultImageIsNeeded: resultImageIsNeeded
    ) else {
        return nil
    }
    return mps_retain(graph)
}

@_cdecl("mps_nn_graph_source_image_count")
public func mps_nn_graph_source_image_count(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let graph: MPSNNGraph = mps_borrow(handle) else { return 0 }
    return graph.sourceImageHandles.count
}

@_cdecl("mps_nn_graph_format")
public func mps_nn_graph_format(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let graph: MPSNNGraph = mps_borrow(handle) else { return 0 }
    return graph.format.rawValue
}

@_cdecl("mps_nn_graph_set_format")
public func mps_nn_graph_set_format(
    _ handle: UnsafeMutableRawPointer?,
    _ formatRaw: UInt
) {
    guard let graph: MPSNNGraph = mps_borrow(handle),
          let format = mps_channel_format(formatRaw)
    else {
        return
    }
    graph.format = format
}

@_cdecl("mps_nn_graph_set_output_state_is_temporary")
public func mps_nn_graph_set_output_state_is_temporary(
    _ handle: UnsafeMutableRawPointer?,
    _ temporary: Bool
) {
    guard let graph: MPSNNGraph = mps_borrow(handle) else { return }
    graph.outputStateIsTemporary = temporary
}

@_cdecl("mps_nn_graph_use_default_destination_image_allocator")
public func mps_nn_graph_use_default_destination_image_allocator(
    _ handle: UnsafeMutableRawPointer?
) {
    guard let graph: MPSNNGraph = mps_borrow(handle) else { return }
    graph.destinationImageAllocator = MPSImage.defaultAllocator()
}

@_cdecl("mps_nn_graph_reload_from_data_sources")
public func mps_nn_graph_reload_from_data_sources(
    _ handle: UnsafeMutableRawPointer?
) {
    guard let graph: MPSNNGraph = mps_borrow(handle) else { return }
    graph.reloadFromDataSources()
}

@_cdecl("mps_nn_graph_encode")
public func mps_nn_graph_encode(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ sourceImageCount: Int,
    _ sourceImageHandles: UnsafePointer<UnsafeMutableRawPointer?>?
) -> UnsafeMutableRawPointer? {
    guard let graph: MPSNNGraph = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let sourceImageHandles,
          let sourceImages = mps_nn_source_images(count: sourceImageCount, handles: sourceImageHandles)
    else {
        return nil
    }

    guard let result = graph.encode(to: commandBuffer, sourceImages: sourceImages) else {
        return nil
    }
    return mps_retain(result)
}

@_cdecl("mps_cnn_convolution_descriptor_new")
public func mps_cnn_convolution_descriptor_new(
    _ kernelWidth: Int,
    _ kernelHeight: Int,
    _ inputFeatureChannels: Int,
    _ outputFeatureChannels: Int
) -> UnsafeMutableRawPointer? {
    mps_retain(
        MPSCNNConvolutionDescriptor(
            kernelWidth: kernelWidth,
            kernelHeight: kernelHeight,
            inputFeatureChannels: inputFeatureChannels,
            outputFeatureChannels: outputFeatureChannels
        )
    )
}

@_cdecl("mps_cnn_convolution_descriptor_kernel_width")
public func mps_cnn_convolution_descriptor_kernel_width(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.kernelWidth
}

@_cdecl("mps_cnn_convolution_descriptor_kernel_height")
public func mps_cnn_convolution_descriptor_kernel_height(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.kernelHeight
}

@_cdecl("mps_cnn_convolution_descriptor_stride_in_pixels_x")
public func mps_cnn_convolution_descriptor_stride_in_pixels_x(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.strideInPixelsX
}

@_cdecl("mps_cnn_convolution_descriptor_set_stride_in_pixels_x")
public func mps_cnn_convolution_descriptor_set_stride_in_pixels_x(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Int
) {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return }
    descriptor.strideInPixelsX = value
}

@_cdecl("mps_cnn_convolution_descriptor_stride_in_pixels_y")
public func mps_cnn_convolution_descriptor_stride_in_pixels_y(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.strideInPixelsY
}

@_cdecl("mps_cnn_convolution_descriptor_set_stride_in_pixels_y")
public func mps_cnn_convolution_descriptor_set_stride_in_pixels_y(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Int
) {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return }
    descriptor.strideInPixelsY = value
}

@_cdecl("mps_cnn_convolution_descriptor_groups")
public func mps_cnn_convolution_descriptor_groups(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.groups
}

@_cdecl("mps_cnn_convolution_descriptor_set_groups")
public func mps_cnn_convolution_descriptor_set_groups(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Int
) {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return }
    descriptor.groups = value
}

@_cdecl("mps_cnn_convolution_descriptor_dilation_rate_x")
public func mps_cnn_convolution_descriptor_dilation_rate_x(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.dilationRateX
}

@_cdecl("mps_cnn_convolution_descriptor_set_dilation_rate_x")
public func mps_cnn_convolution_descriptor_set_dilation_rate_x(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Int
) {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return }
    descriptor.dilationRateX = value
}

@_cdecl("mps_cnn_convolution_descriptor_dilation_rate_y")
public func mps_cnn_convolution_descriptor_dilation_rate_y(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.dilationRateY
}

@_cdecl("mps_cnn_convolution_descriptor_set_dilation_rate_y")
public func mps_cnn_convolution_descriptor_set_dilation_rate_y(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Int
) {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return }
    descriptor.dilationRateY = value
}

@_cdecl("mps_rnn_single_gate_descriptor_new")
public func mps_rnn_single_gate_descriptor_new(
    _ inputFeatureChannels: Int,
    _ outputFeatureChannels: Int
) -> UnsafeMutableRawPointer? {
    mps_retain(
        MPSRNNSingleGateDescriptor.createRNNSingleGateDescriptor(
            withInputFeatureChannels: inputFeatureChannels,
            outputFeatureChannels: outputFeatureChannels
        )
    )
}

@_cdecl("mps_rnn_single_gate_descriptor_input_feature_channels")
public func mps_rnn_single_gate_descriptor_input_feature_channels(
    _ handle: UnsafeMutableRawPointer?
) -> Int {
    guard let descriptor: MPSRNNSingleGateDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.inputFeatureChannels
}

@_cdecl("mps_rnn_single_gate_descriptor_set_input_feature_channels")
public func mps_rnn_single_gate_descriptor_set_input_feature_channels(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Int
) {
    guard let descriptor: MPSRNNSingleGateDescriptor = mps_borrow(handle) else { return }
    descriptor.inputFeatureChannels = value
}

@_cdecl("mps_rnn_single_gate_descriptor_output_feature_channels")
public func mps_rnn_single_gate_descriptor_output_feature_channels(
    _ handle: UnsafeMutableRawPointer?
) -> Int {
    guard let descriptor: MPSRNNSingleGateDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.outputFeatureChannels
}

@_cdecl("mps_rnn_single_gate_descriptor_set_output_feature_channels")
public func mps_rnn_single_gate_descriptor_set_output_feature_channels(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Int
) {
    guard let descriptor: MPSRNNSingleGateDescriptor = mps_borrow(handle) else { return }
    descriptor.outputFeatureChannels = value
}

@_cdecl("mps_rnn_single_gate_descriptor_use_layer_input_unit_transform_mode")
public func mps_rnn_single_gate_descriptor_use_layer_input_unit_transform_mode(
    _ handle: UnsafeMutableRawPointer?
) -> Bool {
    guard let descriptor: MPSRNNSingleGateDescriptor = mps_borrow(handle) else { return false }
    return descriptor.useLayerInputUnitTransformMode
}

@_cdecl("mps_rnn_single_gate_descriptor_set_use_layer_input_unit_transform_mode")
public func mps_rnn_single_gate_descriptor_set_use_layer_input_unit_transform_mode(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Bool
) {
    guard let descriptor: MPSRNNSingleGateDescriptor = mps_borrow(handle) else { return }
    descriptor.useLayerInputUnitTransformMode = value
}

@_cdecl("mps_rnn_single_gate_descriptor_use_float32_weights")
public func mps_rnn_single_gate_descriptor_use_float32_weights(
    _ handle: UnsafeMutableRawPointer?
) -> Bool {
    guard let descriptor: MPSRNNSingleGateDescriptor = mps_borrow(handle) else { return false }
    return descriptor.useFloat32Weights
}

@_cdecl("mps_rnn_single_gate_descriptor_set_use_float32_weights")
public func mps_rnn_single_gate_descriptor_set_use_float32_weights(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Bool
) {
    guard let descriptor: MPSRNNSingleGateDescriptor = mps_borrow(handle) else { return }
    descriptor.useFloat32Weights = value
}

@_cdecl("mps_rnn_single_gate_descriptor_layer_sequence_direction")
public func mps_rnn_single_gate_descriptor_layer_sequence_direction(
    _ handle: UnsafeMutableRawPointer?
) -> UInt {
    guard let descriptor: MPSRNNSingleGateDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.layerSequenceDirection.rawValue
}

@_cdecl("mps_rnn_single_gate_descriptor_set_layer_sequence_direction")
public func mps_rnn_single_gate_descriptor_set_layer_sequence_direction(
    _ handle: UnsafeMutableRawPointer?,
    _ value: UInt
) {
    guard let descriptor: MPSRNNSingleGateDescriptor = mps_borrow(handle),
          let direction = MPSRNNSequenceDirection(rawValue: value)
    else {
        return
    }
    descriptor.layerSequenceDirection = direction
}
