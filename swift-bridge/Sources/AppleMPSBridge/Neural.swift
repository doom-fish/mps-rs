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

@_cdecl("mps_cnn_convolution_descriptor_input_feature_channels")
public func mps_cnn_convolution_descriptor_input_feature_channels(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.inputFeatureChannels
}

@_cdecl("mps_cnn_convolution_descriptor_output_feature_channels")
public func mps_cnn_convolution_descriptor_output_feature_channels(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.outputFeatureChannels
}

@_cdecl("mps_cnn_convolution_new")
public func mps_cnn_convolution_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?,
    _ kernelWeights: UnsafePointer<Float>?,
    _ biasTerms: UnsafePointer<Float>?,
    _ flagsRaw: UInt
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(descriptorHandle),
          let kernelWeights,
          let flags = MPSCNNConvolutionFlags(rawValue: flagsRaw)
    else {
        return nil
    }
    return mps_retain(
        MPSCNNConvolution(
            device: device,
            convolutionDescriptor: descriptor,
            kernelWeights: kernelWeights,
            biasTerms: biasTerms,
            flags: flags
        )
    )
}

@_cdecl("mps_cnn_convolution_input_feature_channels")
public func mps_cnn_convolution_input_feature_channels(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let convolution: MPSCNNConvolution = mps_borrow(handle) else { return 0 }
    return convolution.inputFeatureChannels
}

@_cdecl("mps_cnn_convolution_output_feature_channels")
public func mps_cnn_convolution_output_feature_channels(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let convolution: MPSCNNConvolution = mps_borrow(handle) else { return 0 }
    return convolution.outputFeatureChannels
}

@_cdecl("mps_cnn_convolution_groups")
public func mps_cnn_convolution_groups(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let convolution: MPSCNNConvolution = mps_borrow(handle) else { return 0 }
    return convolution.groups
}

@_cdecl("mps_cnn_convolution_sub_pixel_scale_factor")
public func mps_cnn_convolution_sub_pixel_scale_factor(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let convolution: MPSCNNConvolution = mps_borrow(handle) else { return 0 }
    return convolution.subPixelScaleFactor
}

@_cdecl("mps_cnn_convolution_channel_multiplier")
public func mps_cnn_convolution_channel_multiplier(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let convolution: MPSCNNConvolution = mps_borrow(handle) else { return 0 }
    return convolution.channelMultiplier
}

@_cdecl("mps_cnn_convolution_accumulator_precision_option")
public func mps_cnn_convolution_accumulator_precision_option(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let convolution: MPSCNNConvolution = mps_borrow(handle) else { return 0 }
    return convolution.accumulatorPrecisionOption.rawValue
}

@_cdecl("mps_cnn_convolution_set_accumulator_precision_option")
public func mps_cnn_convolution_set_accumulator_precision_option(
    _ handle: UnsafeMutableRawPointer?,
    _ value: UInt
) {
    guard let convolution: MPSCNNConvolution = mps_borrow(handle) else { return }
    convolution.accumulatorPrecisionOption = MPSNNConvolutionAccumulatorPrecisionOption(rawValue: value)
}

@_cdecl("mps_cnn_convolution_encode_image")
public func mps_cnn_convolution_encode_image(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ sourceImageHandle: UnsafeMutableRawPointer?,
    _ destinationImageHandle: UnsafeMutableRawPointer?
) {
    guard let convolution: MPSCNNConvolution = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let sourceImage: MPSImage = mps_borrow(sourceImageHandle),
          let destinationImage: MPSImage = mps_borrow(destinationImageHandle)
    else {
        return
    }
    convolution.encode(commandBuffer: commandBuffer, sourceImage: sourceImage, destinationImage: destinationImage)
}

@_cdecl("mps_cnn_convolution_weights_and_biases_state_new")
public func mps_cnn_convolution_weights_and_biases_state_new(
    _ weightsHandle: UnsafeMutableRawPointer?,
    _ biasesHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let weights: MTLBuffer = mps_borrow(weightsHandle) else { return nil }
    let biases: MTLBuffer? = mps_borrow(biasesHandle)
    return mps_retain(MPSCNNConvolutionWeightsAndBiasesState(weights: weights, biases: biases))
}

@_cdecl("mps_cnn_convolution_weights_and_biases_state_new_with_offsets")
public func mps_cnn_convolution_weights_and_biases_state_new_with_offsets(
    _ weightsHandle: UnsafeMutableRawPointer?,
    _ weightsOffset: Int,
    _ biasesHandle: UnsafeMutableRawPointer?,
    _ biasesOffset: Int,
    _ descriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let weights: MTLBuffer = mps_borrow(weightsHandle),
          let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(descriptorHandle)
    else {
        return nil
    }
    let biases: MTLBuffer? = mps_borrow(biasesHandle)
    return mps_retain(
        MPSCNNConvolutionWeightsAndBiasesState(
            weights: weights,
            weightsOffset: weightsOffset,
            biases: biases,
            biasesOffset: biasesOffset,
            cnnConvolutionDescriptor: descriptor
        )
    )
}

@_cdecl("mps_cnn_convolution_weights_and_biases_state_new_with_device")
public func mps_cnn_convolution_weights_and_biases_state_new_with_device(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let descriptor: MPSCNNConvolutionDescriptor = mps_borrow(descriptorHandle)
    else {
        return nil
    }
    return mps_retain(MPSCNNConvolutionWeightsAndBiasesState(device: device, cnnConvolutionDescriptor: descriptor))
}

@_cdecl("mps_cnn_convolution_weights_and_biases_state_weights_offset")
public func mps_cnn_convolution_weights_and_biases_state_weights_offset(
    _ handle: UnsafeMutableRawPointer?
) -> Int {
    guard let state: MPSCNNConvolutionWeightsAndBiasesState = mps_borrow(handle) else { return 0 }
    return state.weightsOffset
}

@_cdecl("mps_cnn_convolution_weights_and_biases_state_biases_offset")
public func mps_cnn_convolution_weights_and_biases_state_biases_offset(
    _ handle: UnsafeMutableRawPointer?
) -> Int {
    guard let state: MPSCNNConvolutionWeightsAndBiasesState = mps_borrow(handle) else { return 0 }
    return state.biasesOffset
}

@_cdecl("mps_nn_optimizer_descriptor_new")
public func mps_nn_optimizer_descriptor_new(
    _ learningRate: Float,
    _ gradientRescale: Float,
    _ regularizationTypeRaw: UInt,
    _ regularizationScale: Float
) -> UnsafeMutableRawPointer? {
    guard let regularizationType = MPSNNRegularizationType(rawValue: regularizationTypeRaw) else {
        return nil
    }
    return mps_retain(
        MPSNNOptimizerDescriptor(
            learningRate: learningRate,
            gradientRescale: gradientRescale,
            regularizationType: regularizationType,
            regularizationScale: regularizationScale
        )
    )
}

@_cdecl("mps_nn_optimizer_descriptor_new_with_gradient_clipping")
public func mps_nn_optimizer_descriptor_new_with_gradient_clipping(
    _ learningRate: Float,
    _ gradientRescale: Float,
    _ applyGradientClipping: Bool,
    _ gradientClipMax: Float,
    _ gradientClipMin: Float,
    _ regularizationTypeRaw: UInt,
    _ regularizationScale: Float
) -> UnsafeMutableRawPointer? {
    guard let regularizationType = MPSNNRegularizationType(rawValue: regularizationTypeRaw) else {
        return nil
    }
    return mps_retain(
        MPSNNOptimizerDescriptor(
            learningRate: learningRate,
            gradientRescale: gradientRescale,
            applyGradientClipping: applyGradientClipping,
            gradientClipMax: gradientClipMax,
            gradientClipMin: gradientClipMin,
            regularizationType: regularizationType,
            regularizationScale: regularizationScale
        )
    )
}

@_cdecl("mps_nn_optimizer_descriptor_learning_rate")
public func mps_nn_optimizer_descriptor_learning_rate(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return 0.0 }
    return descriptor.learningRate
}

@_cdecl("mps_nn_optimizer_descriptor_set_learning_rate")
public func mps_nn_optimizer_descriptor_set_learning_rate(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Float
) {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return }
    descriptor.learningRate = value
}

@_cdecl("mps_nn_optimizer_descriptor_gradient_rescale")
public func mps_nn_optimizer_descriptor_gradient_rescale(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return 0.0 }
    return descriptor.gradientRescale
}

@_cdecl("mps_nn_optimizer_descriptor_set_gradient_rescale")
public func mps_nn_optimizer_descriptor_set_gradient_rescale(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Float
) {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return }
    descriptor.gradientRescale = value
}

@_cdecl("mps_nn_optimizer_descriptor_apply_gradient_clipping")
public func mps_nn_optimizer_descriptor_apply_gradient_clipping(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return false }
    return descriptor.applyGradientClipping
}

@_cdecl("mps_nn_optimizer_descriptor_set_apply_gradient_clipping")
public func mps_nn_optimizer_descriptor_set_apply_gradient_clipping(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Bool
) {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return }
    descriptor.applyGradientClipping = value
}

@_cdecl("mps_nn_optimizer_descriptor_gradient_clip_max")
public func mps_nn_optimizer_descriptor_gradient_clip_max(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return 0.0 }
    return descriptor.gradientClipMax
}

@_cdecl("mps_nn_optimizer_descriptor_set_gradient_clip_max")
public func mps_nn_optimizer_descriptor_set_gradient_clip_max(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Float
) {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return }
    descriptor.gradientClipMax = value
}

@_cdecl("mps_nn_optimizer_descriptor_gradient_clip_min")
public func mps_nn_optimizer_descriptor_gradient_clip_min(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return 0.0 }
    return descriptor.gradientClipMin
}

@_cdecl("mps_nn_optimizer_descriptor_set_gradient_clip_min")
public func mps_nn_optimizer_descriptor_set_gradient_clip_min(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Float
) {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return }
    descriptor.gradientClipMin = value
}

@_cdecl("mps_nn_optimizer_descriptor_regularization_scale")
public func mps_nn_optimizer_descriptor_regularization_scale(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return 0.0 }
    return descriptor.regularizationScale
}

@_cdecl("mps_nn_optimizer_descriptor_set_regularization_scale")
public func mps_nn_optimizer_descriptor_set_regularization_scale(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Float
) {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return }
    descriptor.regularizationScale = value
}

@_cdecl("mps_nn_optimizer_descriptor_regularization_type")
public func mps_nn_optimizer_descriptor_regularization_type(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.regularizationType.rawValue
}

@_cdecl("mps_nn_optimizer_descriptor_set_regularization_type")
public func mps_nn_optimizer_descriptor_set_regularization_type(
    _ handle: UnsafeMutableRawPointer?,
    _ value: UInt
) {
    guard let descriptor: MPSNNOptimizerDescriptor = mps_borrow(handle),
          let regularizationType = MPSNNRegularizationType(rawValue: value)
    else {
        return
    }
    descriptor.regularizationType = regularizationType
}

@_cdecl("mps_nn_optimizer_learning_rate")
public func mps_nn_optimizer_learning_rate(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let optimizer: MPSNNOptimizer = mps_borrow(handle) else { return 0.0 }
    return optimizer.learningRate
}

@_cdecl("mps_nn_optimizer_set_learning_rate")
public func mps_nn_optimizer_set_learning_rate(_ handle: UnsafeMutableRawPointer?, _ value: Float) {
    guard let optimizer: MPSNNOptimizer = mps_borrow(handle) else { return }
    optimizer.setLearningRate(value)
}

@_cdecl("mps_nn_optimizer_gradient_rescale")
public func mps_nn_optimizer_gradient_rescale(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let optimizer: MPSNNOptimizer = mps_borrow(handle) else { return 0.0 }
    return optimizer.gradientRescale
}

@_cdecl("mps_nn_optimizer_apply_gradient_clipping")
public func mps_nn_optimizer_apply_gradient_clipping(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let optimizer: MPSNNOptimizer = mps_borrow(handle) else { return false }
    return optimizer.applyGradientClipping
}

@_cdecl("mps_nn_optimizer_set_apply_gradient_clipping")
public func mps_nn_optimizer_set_apply_gradient_clipping(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Bool
) {
    guard let optimizer: MPSNNOptimizer = mps_borrow(handle) else { return }
    optimizer.applyGradientClipping = value
}

@_cdecl("mps_nn_optimizer_gradient_clip_max")
public func mps_nn_optimizer_gradient_clip_max(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let optimizer: MPSNNOptimizer = mps_borrow(handle) else { return 0.0 }
    return optimizer.gradientClipMax
}

@_cdecl("mps_nn_optimizer_gradient_clip_min")
public func mps_nn_optimizer_gradient_clip_min(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let optimizer: MPSNNOptimizer = mps_borrow(handle) else { return 0.0 }
    return optimizer.gradientClipMin
}

@_cdecl("mps_nn_optimizer_regularization_scale")
public func mps_nn_optimizer_regularization_scale(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let optimizer: MPSNNOptimizer = mps_borrow(handle) else { return 0.0 }
    return optimizer.regularizationScale
}

@_cdecl("mps_nn_optimizer_regularization_type")
public func mps_nn_optimizer_regularization_type(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let optimizer: MPSNNOptimizer = mps_borrow(handle) else { return 0 }
    return optimizer.regularizationType.rawValue
}

@_cdecl("mps_nn_optimizer_sgd_new")
public func mps_nn_optimizer_sgd_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ learningRate: Float
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSNNOptimizerStochasticGradientDescent(device: device, learningRate: learningRate))
}

@_cdecl("mps_nn_optimizer_sgd_new_with_options")
public func mps_nn_optimizer_sgd_new_with_options(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ momentumScale: Float,
    _ useNesterovMomentum: Bool,
    _ optimizerDescriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let optimizerDescriptor: MPSNNOptimizerDescriptor = mps_borrow(optimizerDescriptorHandle)
    else {
        return nil
    }
    return mps_retain(
        MPSNNOptimizerStochasticGradientDescent(
            device: device,
            momentumScale: momentumScale,
            useNesterovMomentum: useNesterovMomentum,
            optimizerDescriptor: optimizerDescriptor
        )
    )
}

@_cdecl("mps_nn_optimizer_sgd_momentum_scale")
public func mps_nn_optimizer_sgd_momentum_scale(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let optimizer: MPSNNOptimizerStochasticGradientDescent = mps_borrow(handle) else { return 0.0 }
    return optimizer.momentumScale
}

@_cdecl("mps_nn_optimizer_sgd_use_nesterov_momentum")
public func mps_nn_optimizer_sgd_use_nesterov_momentum(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let optimizer: MPSNNOptimizerStochasticGradientDescent = mps_borrow(handle) else { return false }
    return optimizer.useNesterovMomentum
}

@_cdecl("mps_nn_optimizer_sgd_encode_vector")
public func mps_nn_optimizer_sgd_encode_vector(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ inputGradientVectorHandle: UnsafeMutableRawPointer?,
    _ inputValuesVectorHandle: UnsafeMutableRawPointer?,
    _ inputMomentumVectorHandle: UnsafeMutableRawPointer?,
    _ resultValuesVectorHandle: UnsafeMutableRawPointer?
) {
    guard let optimizer: MPSNNOptimizerStochasticGradientDescent = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let inputGradientVector: MPSVector = mps_borrow(inputGradientVectorHandle),
          let inputValuesVector: MPSVector = mps_borrow(inputValuesVectorHandle),
          let resultValuesVector: MPSVector = mps_borrow(resultValuesVectorHandle)
    else {
        return
    }
    let inputMomentumVector: MPSVector? = mps_borrow(inputMomentumVectorHandle)
    optimizer.encode(
        commandBuffer: commandBuffer,
        inputGradientVector: inputGradientVector,
        inputValuesVector: inputValuesVector,
        inputMomentumVector: inputMomentumVector,
        resultValuesVector: resultValuesVector
    )
}

@_cdecl("mps_nn_optimizer_sgd_encode_matrix")
public func mps_nn_optimizer_sgd_encode_matrix(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ inputGradientMatrixHandle: UnsafeMutableRawPointer?,
    _ inputValuesMatrixHandle: UnsafeMutableRawPointer?,
    _ inputMomentumMatrixHandle: UnsafeMutableRawPointer?,
    _ resultValuesMatrixHandle: UnsafeMutableRawPointer?
) {
    guard let optimizer: MPSNNOptimizerStochasticGradientDescent = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let inputGradientMatrix: MPSMatrix = mps_borrow(inputGradientMatrixHandle),
          let inputValuesMatrix: MPSMatrix = mps_borrow(inputValuesMatrixHandle),
          let resultValuesMatrix: MPSMatrix = mps_borrow(resultValuesMatrixHandle)
    else {
        return
    }
    let inputMomentumMatrix: MPSMatrix? = mps_borrow(inputMomentumMatrixHandle)
    optimizer.encode(
        commandBuffer: commandBuffer,
        inputGradientMatrix: inputGradientMatrix,
        inputValuesMatrix: inputValuesMatrix,
        inputMomentumMatrix: inputMomentumMatrix,
        resultValuesMatrix: resultValuesMatrix
    )
}

@_cdecl("mps_nn_optimizer_rmsprop_new")
public func mps_nn_optimizer_rmsprop_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ learningRate: Float
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSNNOptimizerRMSProp(device: device, learningRate: learningRate))
}

@_cdecl("mps_nn_optimizer_rmsprop_new_with_options")
public func mps_nn_optimizer_rmsprop_new_with_options(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ decay: Double,
    _ epsilon: Float,
    _ optimizerDescriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let optimizerDescriptor: MPSNNOptimizerDescriptor = mps_borrow(optimizerDescriptorHandle)
    else {
        return nil
    }
    return mps_retain(
        MPSNNOptimizerRMSProp(
            device: device,
            decay: decay,
            epsilon: epsilon,
            optimizerDescriptor: optimizerDescriptor
        )
    )
}

@_cdecl("mps_nn_optimizer_rmsprop_decay")
public func mps_nn_optimizer_rmsprop_decay(_ handle: UnsafeMutableRawPointer?) -> Double {
    guard let optimizer: MPSNNOptimizerRMSProp = mps_borrow(handle) else { return 0.0 }
    return optimizer.decay
}

@_cdecl("mps_nn_optimizer_rmsprop_epsilon")
public func mps_nn_optimizer_rmsprop_epsilon(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let optimizer: MPSNNOptimizerRMSProp = mps_borrow(handle) else { return 0.0 }
    return optimizer.epsilon
}

@_cdecl("mps_nn_optimizer_rmsprop_encode_vector")
public func mps_nn_optimizer_rmsprop_encode_vector(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ inputGradientVectorHandle: UnsafeMutableRawPointer?,
    _ inputValuesVectorHandle: UnsafeMutableRawPointer?,
    _ inputSumOfSquaresVectorHandle: UnsafeMutableRawPointer?,
    _ resultValuesVectorHandle: UnsafeMutableRawPointer?
) {
    guard let optimizer: MPSNNOptimizerRMSProp = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let inputGradientVector: MPSVector = mps_borrow(inputGradientVectorHandle),
          let inputValuesVector: MPSVector = mps_borrow(inputValuesVectorHandle),
          let inputSumOfSquaresVector: MPSVector = mps_borrow(inputSumOfSquaresVectorHandle),
          let resultValuesVector: MPSVector = mps_borrow(resultValuesVectorHandle)
    else {
        return
    }
    optimizer.encode(
        commandBuffer: commandBuffer,
        inputGradientVector: inputGradientVector,
        inputValuesVector: inputValuesVector,
        inputSumOfSquaresVector: inputSumOfSquaresVector,
        resultValuesVector: resultValuesVector
    )
}

@_cdecl("mps_nn_optimizer_rmsprop_encode_matrix")
public func mps_nn_optimizer_rmsprop_encode_matrix(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ inputGradientMatrixHandle: UnsafeMutableRawPointer?,
    _ inputValuesMatrixHandle: UnsafeMutableRawPointer?,
    _ inputSumOfSquaresMatrixHandle: UnsafeMutableRawPointer?,
    _ resultValuesMatrixHandle: UnsafeMutableRawPointer?
) {
    guard let optimizer: MPSNNOptimizerRMSProp = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let inputGradientMatrix: MPSMatrix = mps_borrow(inputGradientMatrixHandle),
          let inputValuesMatrix: MPSMatrix = mps_borrow(inputValuesMatrixHandle),
          let inputSumOfSquaresMatrix: MPSMatrix = mps_borrow(inputSumOfSquaresMatrixHandle),
          let resultValuesMatrix: MPSMatrix = mps_borrow(resultValuesMatrixHandle)
    else {
        return
    }
    optimizer.encode(
        commandBuffer: commandBuffer,
        inputGradientMatrix: inputGradientMatrix,
        inputValuesMatrix: inputValuesMatrix,
        inputSumOfSquaresMatrix: inputSumOfSquaresMatrix,
        resultValuesMatrix: resultValuesMatrix
    )
}

@_cdecl("mps_nn_optimizer_adam_new")
public func mps_nn_optimizer_adam_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ learningRate: Float
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSNNOptimizerAdam(device: device, learningRate: learningRate))
}

@_cdecl("mps_nn_optimizer_adam_new_with_options")
public func mps_nn_optimizer_adam_new_with_options(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ beta1: Double,
    _ beta2: Double,
    _ epsilon: Float,
    _ timeStep: Int,
    _ optimizerDescriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let optimizerDescriptor: MPSNNOptimizerDescriptor = mps_borrow(optimizerDescriptorHandle)
    else {
        return nil
    }
    return mps_retain(
        MPSNNOptimizerAdam(
            device: device,
            beta1: beta1,
            beta2: beta2,
            epsilon: epsilon,
            timeStep: timeStep,
            optimizerDescriptor: optimizerDescriptor
        )
    )
}

@_cdecl("mps_nn_optimizer_adam_beta1")
public func mps_nn_optimizer_adam_beta1(_ handle: UnsafeMutableRawPointer?) -> Double {
    guard let optimizer: MPSNNOptimizerAdam = mps_borrow(handle) else { return 0.0 }
    return optimizer.beta1
}

@_cdecl("mps_nn_optimizer_adam_beta2")
public func mps_nn_optimizer_adam_beta2(_ handle: UnsafeMutableRawPointer?) -> Double {
    guard let optimizer: MPSNNOptimizerAdam = mps_borrow(handle) else { return 0.0 }
    return optimizer.beta2
}

@_cdecl("mps_nn_optimizer_adam_epsilon")
public func mps_nn_optimizer_adam_epsilon(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let optimizer: MPSNNOptimizerAdam = mps_borrow(handle) else { return 0.0 }
    return optimizer.epsilon
}

@_cdecl("mps_nn_optimizer_adam_time_step")
public func mps_nn_optimizer_adam_time_step(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let optimizer: MPSNNOptimizerAdam = mps_borrow(handle) else { return 0 }
    return optimizer.timeStep
}

@_cdecl("mps_nn_optimizer_adam_set_time_step")
public func mps_nn_optimizer_adam_set_time_step(_ handle: UnsafeMutableRawPointer?, _ value: Int) {
    guard let optimizer: MPSNNOptimizerAdam = mps_borrow(handle) else { return }
    optimizer.timeStep = value
}

@_cdecl("mps_nn_optimizer_adam_encode_vector")
public func mps_nn_optimizer_adam_encode_vector(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ inputGradientVectorHandle: UnsafeMutableRawPointer?,
    _ inputValuesVectorHandle: UnsafeMutableRawPointer?,
    _ inputMomentumVectorHandle: UnsafeMutableRawPointer?,
    _ inputVelocityVectorHandle: UnsafeMutableRawPointer?,
    _ resultValuesVectorHandle: UnsafeMutableRawPointer?
) {
    guard let optimizer: MPSNNOptimizerAdam = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let inputGradientVector: MPSVector = mps_borrow(inputGradientVectorHandle),
          let inputValuesVector: MPSVector = mps_borrow(inputValuesVectorHandle),
          let inputMomentumVector: MPSVector = mps_borrow(inputMomentumVectorHandle),
          let inputVelocityVector: MPSVector = mps_borrow(inputVelocityVectorHandle),
          let resultValuesVector: MPSVector = mps_borrow(resultValuesVectorHandle)
    else {
        return
    }
    optimizer.encode(
        commandBuffer: commandBuffer,
        inputGradientVector: inputGradientVector,
        inputValuesVector: inputValuesVector,
        inputMomentumVector: inputMomentumVector,
        inputVelocityVector: inputVelocityVector,
        resultValuesVector: resultValuesVector
    )
}

@_cdecl("mps_nn_optimizer_adam_encode_matrix")
public func mps_nn_optimizer_adam_encode_matrix(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ inputGradientMatrixHandle: UnsafeMutableRawPointer?,
    _ inputValuesMatrixHandle: UnsafeMutableRawPointer?,
    _ inputMomentumMatrixHandle: UnsafeMutableRawPointer?,
    _ inputVelocityMatrixHandle: UnsafeMutableRawPointer?,
    _ resultValuesMatrixHandle: UnsafeMutableRawPointer?
) {
    guard let optimizer: MPSNNOptimizerAdam = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let inputGradientMatrix: MPSMatrix = mps_borrow(inputGradientMatrixHandle),
          let inputValuesMatrix: MPSMatrix = mps_borrow(inputValuesMatrixHandle),
          let inputMomentumMatrix: MPSMatrix = mps_borrow(inputMomentumMatrixHandle),
          let inputVelocityMatrix: MPSMatrix = mps_borrow(inputVelocityMatrixHandle),
          let resultValuesMatrix: MPSMatrix = mps_borrow(resultValuesMatrixHandle)
    else {
        return
    }
    optimizer.encode(
        commandBuffer: commandBuffer,
        inputGradientMatrix: inputGradientMatrix,
        inputValuesMatrix: inputValuesMatrix,
        inputMomentumMatrix: inputMomentumMatrix,
        inputVelocityMatrix: inputVelocityMatrix,
        resultValuesMatrix: resultValuesMatrix
    )
}

@_cdecl("mps_nn_optimizer_adam_encode_amsgrad_vector")
public func mps_nn_optimizer_adam_encode_amsgrad_vector(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ inputGradientVectorHandle: UnsafeMutableRawPointer?,
    _ inputValuesVectorHandle: UnsafeMutableRawPointer?,
    _ inputMomentumVectorHandle: UnsafeMutableRawPointer?,
    _ inputVelocityVectorHandle: UnsafeMutableRawPointer?,
    _ maximumVelocityVectorHandle: UnsafeMutableRawPointer?,
    _ resultValuesVectorHandle: UnsafeMutableRawPointer?
) {
    guard let optimizer: MPSNNOptimizerAdam = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let inputGradientVector: MPSVector = mps_borrow(inputGradientVectorHandle),
          let inputValuesVector: MPSVector = mps_borrow(inputValuesVectorHandle),
          let inputMomentumVector: MPSVector = mps_borrow(inputMomentumVectorHandle),
          let inputVelocityVector: MPSVector = mps_borrow(inputVelocityVectorHandle),
          let resultValuesVector: MPSVector = mps_borrow(resultValuesVectorHandle)
    else {
        return
    }
    let maximumVelocityVector: MPSVector? = mps_borrow(maximumVelocityVectorHandle)
    optimizer.encode(
        commandBuffer: commandBuffer,
        inputGradientVector: inputGradientVector,
        inputValuesVector: inputValuesVector,
        inputMomentumVector: inputMomentumVector,
        inputVelocityVector: inputVelocityVector,
        maximumVelocityVector: maximumVelocityVector,
        resultValuesVector: resultValuesVector
    )
}

@_cdecl("mps_nn_optimizer_adam_encode_amsgrad_matrix")
public func mps_nn_optimizer_adam_encode_amsgrad_matrix(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ inputGradientMatrixHandle: UnsafeMutableRawPointer?,
    _ inputValuesMatrixHandle: UnsafeMutableRawPointer?,
    _ inputMomentumMatrixHandle: UnsafeMutableRawPointer?,
    _ inputVelocityMatrixHandle: UnsafeMutableRawPointer?,
    _ maximumVelocityMatrixHandle: UnsafeMutableRawPointer?,
    _ resultValuesMatrixHandle: UnsafeMutableRawPointer?
) {
    guard let optimizer: MPSNNOptimizerAdam = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let inputGradientMatrix: MPSMatrix = mps_borrow(inputGradientMatrixHandle),
          let inputValuesMatrix: MPSMatrix = mps_borrow(inputValuesMatrixHandle),
          let inputMomentumMatrix: MPSMatrix = mps_borrow(inputMomentumMatrixHandle),
          let inputVelocityMatrix: MPSMatrix = mps_borrow(inputVelocityMatrixHandle),
          let resultValuesMatrix: MPSMatrix = mps_borrow(resultValuesMatrixHandle)
    else {
        return
    }
    let maximumVelocityMatrix: MPSMatrix? = mps_borrow(maximumVelocityMatrixHandle)
    optimizer.encode(
        commandBuffer: commandBuffer,
        inputGradientMatrix: inputGradientMatrix,
        inputValuesMatrix: inputValuesMatrix,
        inputMomentumMatrix: inputMomentumMatrix,
        inputVelocityMatrix: inputVelocityMatrix,
        maximumVelocityMatrix: maximumVelocityMatrix,
        resultValuesMatrix: resultValuesMatrix
    )
}

@_cdecl("mps_rnn_descriptor_input_feature_channels")
public func mps_rnn_descriptor_input_feature_channels(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let descriptor: MPSRNNDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.inputFeatureChannels
}

@_cdecl("mps_rnn_descriptor_set_input_feature_channels")
public func mps_rnn_descriptor_set_input_feature_channels(_ handle: UnsafeMutableRawPointer?, _ value: Int) {
    guard let descriptor: MPSRNNDescriptor = mps_borrow(handle) else { return }
    descriptor.inputFeatureChannels = value
}

@_cdecl("mps_rnn_descriptor_output_feature_channels")
public func mps_rnn_descriptor_output_feature_channels(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let descriptor: MPSRNNDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.outputFeatureChannels
}

@_cdecl("mps_rnn_descriptor_set_output_feature_channels")
public func mps_rnn_descriptor_set_output_feature_channels(_ handle: UnsafeMutableRawPointer?, _ value: Int) {
    guard let descriptor: MPSRNNDescriptor = mps_borrow(handle) else { return }
    descriptor.outputFeatureChannels = value
}

@_cdecl("mps_rnn_descriptor_use_layer_input_unit_transform_mode")
public func mps_rnn_descriptor_use_layer_input_unit_transform_mode(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let descriptor: MPSRNNDescriptor = mps_borrow(handle) else { return false }
    return descriptor.useLayerInputUnitTransformMode
}

@_cdecl("mps_rnn_descriptor_set_use_layer_input_unit_transform_mode")
public func mps_rnn_descriptor_set_use_layer_input_unit_transform_mode(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Bool
) {
    guard let descriptor: MPSRNNDescriptor = mps_borrow(handle) else { return }
    descriptor.useLayerInputUnitTransformMode = value
}

@_cdecl("mps_rnn_descriptor_use_float32_weights")
public func mps_rnn_descriptor_use_float32_weights(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let descriptor: MPSRNNDescriptor = mps_borrow(handle) else { return false }
    return descriptor.useFloat32Weights
}

@_cdecl("mps_rnn_descriptor_set_use_float32_weights")
public func mps_rnn_descriptor_set_use_float32_weights(_ handle: UnsafeMutableRawPointer?, _ value: Bool) {
    guard let descriptor: MPSRNNDescriptor = mps_borrow(handle) else { return }
    descriptor.useFloat32Weights = value
}

@_cdecl("mps_rnn_descriptor_layer_sequence_direction")
public func mps_rnn_descriptor_layer_sequence_direction(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let descriptor: MPSRNNDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.layerSequenceDirection.rawValue
}

@_cdecl("mps_rnn_descriptor_set_layer_sequence_direction")
public func mps_rnn_descriptor_set_layer_sequence_direction(_ handle: UnsafeMutableRawPointer?, _ value: UInt) {
    guard let descriptor: MPSRNNDescriptor = mps_borrow(handle),
          let direction = MPSRNNSequenceDirection(rawValue: value)
    else {
        return
    }
    descriptor.layerSequenceDirection = direction
}

@_cdecl("mps_gru_descriptor_new")
public func mps_gru_descriptor_new(
    _ inputFeatureChannels: Int,
    _ outputFeatureChannels: Int
) -> UnsafeMutableRawPointer? {
    mps_retain(
        MPSGRUDescriptor.createGRUDescriptor(
            withInputFeatureChannels: inputFeatureChannels,
            outputFeatureChannels: outputFeatureChannels
        )
    )
}

@_cdecl("mps_gru_descriptor_gate_pnorm_value")
public func mps_gru_descriptor_gate_pnorm_value(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let descriptor: MPSGRUDescriptor = mps_borrow(handle) else { return 0.0 }
    return descriptor.gatePnormValue
}

@_cdecl("mps_gru_descriptor_set_gate_pnorm_value")
public func mps_gru_descriptor_set_gate_pnorm_value(_ handle: UnsafeMutableRawPointer?, _ value: Float) {
    guard let descriptor: MPSGRUDescriptor = mps_borrow(handle) else { return }
    descriptor.gatePnormValue = value
}

@_cdecl("mps_gru_descriptor_flip_output_gates")
public func mps_gru_descriptor_flip_output_gates(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let descriptor: MPSGRUDescriptor = mps_borrow(handle) else { return false }
    return descriptor.flipOutputGates
}

@_cdecl("mps_gru_descriptor_set_flip_output_gates")
public func mps_gru_descriptor_set_flip_output_gates(_ handle: UnsafeMutableRawPointer?, _ value: Bool) {
    guard let descriptor: MPSGRUDescriptor = mps_borrow(handle) else { return }
    descriptor.flipOutputGates = value
}

@_cdecl("mps_lstm_descriptor_new")
public func mps_lstm_descriptor_new(
    _ inputFeatureChannels: Int,
    _ outputFeatureChannels: Int
) -> UnsafeMutableRawPointer? {
    mps_retain(
        MPSLSTMDescriptor.createLSTMDescriptor(
            withInputFeatureChannels: inputFeatureChannels,
            outputFeatureChannels: outputFeatureChannels
        )
    )
}

@_cdecl("mps_lstm_descriptor_memory_weights_are_diagonal")
public func mps_lstm_descriptor_memory_weights_are_diagonal(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let descriptor: MPSLSTMDescriptor = mps_borrow(handle) else { return false }
    return descriptor.memoryWeightsAreDiagonal
}

@_cdecl("mps_lstm_descriptor_set_memory_weights_are_diagonal")
public func mps_lstm_descriptor_set_memory_weights_are_diagonal(_ handle: UnsafeMutableRawPointer?, _ value: Bool) {
    guard let descriptor: MPSLSTMDescriptor = mps_borrow(handle) else { return }
    descriptor.memoryWeightsAreDiagonal = value
}

@_cdecl("mps_lstm_descriptor_cell_to_output_neuron_type")
public func mps_lstm_descriptor_cell_to_output_neuron_type(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let descriptor: MPSLSTMDescriptor = mps_borrow(handle) else { return 0 }
    return UInt(descriptor.cellToOutputNeuronType.rawValue)
}

@_cdecl("mps_lstm_descriptor_set_cell_to_output_neuron_type")
public func mps_lstm_descriptor_set_cell_to_output_neuron_type(_ handle: UnsafeMutableRawPointer?, _ value: UInt) {
    guard let descriptor: MPSLSTMDescriptor = mps_borrow(handle),
          let neuronType = MPSCNNNeuronType(rawValue: Int32(value))
    else {
        return
    }
    descriptor.cellToOutputNeuronType = neuronType
}

@_cdecl("mps_lstm_descriptor_cell_to_output_neuron_param_a")
public func mps_lstm_descriptor_cell_to_output_neuron_param_a(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let descriptor: MPSLSTMDescriptor = mps_borrow(handle) else { return 0.0 }
    return descriptor.cellToOutputNeuronParamA
}

@_cdecl("mps_lstm_descriptor_set_cell_to_output_neuron_param_a")
public func mps_lstm_descriptor_set_cell_to_output_neuron_param_a(_ handle: UnsafeMutableRawPointer?, _ value: Float) {
    guard let descriptor: MPSLSTMDescriptor = mps_borrow(handle) else { return }
    descriptor.cellToOutputNeuronParamA = value
}

@_cdecl("mps_lstm_descriptor_cell_to_output_neuron_param_b")
public func mps_lstm_descriptor_cell_to_output_neuron_param_b(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let descriptor: MPSLSTMDescriptor = mps_borrow(handle) else { return 0.0 }
    return descriptor.cellToOutputNeuronParamB
}

@_cdecl("mps_lstm_descriptor_set_cell_to_output_neuron_param_b")
public func mps_lstm_descriptor_set_cell_to_output_neuron_param_b(_ handle: UnsafeMutableRawPointer?, _ value: Float) {
    guard let descriptor: MPSLSTMDescriptor = mps_borrow(handle) else { return }
    descriptor.cellToOutputNeuronParamB = value
}

@_cdecl("mps_lstm_descriptor_cell_to_output_neuron_param_c")
public func mps_lstm_descriptor_cell_to_output_neuron_param_c(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let descriptor: MPSLSTMDescriptor = mps_borrow(handle) else { return 0.0 }
    return descriptor.cellToOutputNeuronParamC
}

@_cdecl("mps_lstm_descriptor_set_cell_to_output_neuron_param_c")
public func mps_lstm_descriptor_set_cell_to_output_neuron_param_c(_ handle: UnsafeMutableRawPointer?, _ value: Float) {
    guard let descriptor: MPSLSTMDescriptor = mps_borrow(handle) else { return }
    descriptor.cellToOutputNeuronParamC = value
}

@_cdecl("mps_rnn_recurrent_image_state_recurrent_output_image")
public func mps_rnn_recurrent_image_state_recurrent_output_image(
    _ handle: UnsafeMutableRawPointer?,
    _ layerIndex: Int
) -> UnsafeMutableRawPointer? {
    guard let state: MPSRNNRecurrentImageState = mps_borrow(handle),
          let image = state.getRecurrentOutputImage(forLayerIndex: layerIndex)
    else {
        return nil
    }
    return mps_retain(image)
}

@_cdecl("mps_rnn_recurrent_image_state_memory_cell_image")
public func mps_rnn_recurrent_image_state_memory_cell_image(
    _ handle: UnsafeMutableRawPointer?,
    _ layerIndex: Int
) -> UnsafeMutableRawPointer? {
    guard let state: MPSRNNRecurrentImageState = mps_borrow(handle),
          let image = state.getMemoryCellImage(forLayerIndex: layerIndex)
    else {
        return nil
    }
    return mps_retain(image)
}

@_cdecl("mps_rnn_image_inference_layer_new")
public func mps_rnn_image_inference_layer_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let descriptor: MPSRNNDescriptor = mps_borrow(descriptorHandle)
    else {
        return nil
    }
    return mps_retain(MPSRNNImageInferenceLayer(device: device, rnnDescriptor: descriptor))
}

@_cdecl("mps_rnn_image_inference_layer_new_stack")
public func mps_rnn_image_inference_layer_new_stack(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ descriptorCount: Int,
    _ descriptorHandles: UnsafePointer<UnsafeMutableRawPointer?>?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let descriptors: [MPSRNNDescriptor] = mps_borrow_array(descriptorHandles, count: descriptorCount)
    else {
        return nil
    }
    return mps_retain(MPSRNNImageInferenceLayer(device: device, rnnDescriptors: descriptors))
}

@_cdecl("mps_rnn_image_inference_layer_input_feature_channels")
public func mps_rnn_image_inference_layer_input_feature_channels(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let layer: MPSRNNImageInferenceLayer = mps_borrow(handle) else { return 0 }
    return layer.inputFeatureChannels
}

@_cdecl("mps_rnn_image_inference_layer_output_feature_channels")
public func mps_rnn_image_inference_layer_output_feature_channels(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let layer: MPSRNNImageInferenceLayer = mps_borrow(handle) else { return 0 }
    return layer.outputFeatureChannels
}

@_cdecl("mps_rnn_image_inference_layer_number_of_layers")
public func mps_rnn_image_inference_layer_number_of_layers(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let layer: MPSRNNImageInferenceLayer = mps_borrow(handle) else { return 0 }
    return layer.numberOfLayers
}

@_cdecl("mps_rnn_image_inference_layer_recurrent_output_is_temporary")
public func mps_rnn_image_inference_layer_recurrent_output_is_temporary(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let layer: MPSRNNImageInferenceLayer = mps_borrow(handle) else { return false }
    return layer.recurrentOutputIsTemporary
}

@_cdecl("mps_rnn_image_inference_layer_set_recurrent_output_is_temporary")
public func mps_rnn_image_inference_layer_set_recurrent_output_is_temporary(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Bool
) {
    guard let layer: MPSRNNImageInferenceLayer = mps_borrow(handle) else { return }
    layer.recurrentOutputIsTemporary = value
}

@_cdecl("mps_rnn_image_inference_layer_store_all_intermediate_states")
public func mps_rnn_image_inference_layer_store_all_intermediate_states(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let layer: MPSRNNImageInferenceLayer = mps_borrow(handle) else { return false }
    return layer.storeAllIntermediateStates
}

@_cdecl("mps_rnn_image_inference_layer_set_store_all_intermediate_states")
public func mps_rnn_image_inference_layer_set_store_all_intermediate_states(
    _ handle: UnsafeMutableRawPointer?,
    _ value: Bool
) {
    guard let layer: MPSRNNImageInferenceLayer = mps_borrow(handle) else { return }
    layer.storeAllIntermediateStates = value
}

@_cdecl("mps_rnn_image_inference_layer_bidirectional_combine_mode")
public func mps_rnn_image_inference_layer_bidirectional_combine_mode(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let layer: MPSRNNImageInferenceLayer = mps_borrow(handle) else { return 0 }
    return layer.bidirectionalCombineMode.rawValue
}

@_cdecl("mps_rnn_image_inference_layer_set_bidirectional_combine_mode")
public func mps_rnn_image_inference_layer_set_bidirectional_combine_mode(
    _ handle: UnsafeMutableRawPointer?,
    _ value: UInt
) {
    guard let layer: MPSRNNImageInferenceLayer = mps_borrow(handle),
          let mode = MPSRNNBidirectionalCombineMode(rawValue: value)
    else {
        return
    }
    layer.bidirectionalCombineMode = mode
}

@_cdecl("mps_rnn_image_inference_layer_encode_sequence")
public func mps_rnn_image_inference_layer_encode_sequence(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ imageCount: Int,
    _ sourceImageHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ destinationImageHandles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ recurrentInputStateHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let layer: MPSRNNImageInferenceLayer = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let sourceImages: [MPSImage] = mps_borrow_array(sourceImageHandles, count: imageCount),
          let destinationImages: [MPSImage] = mps_borrow_array(destinationImageHandles, count: imageCount),
          sourceImages.count == destinationImages.count
    else {
        return nil
    }
    let recurrentInputState: MPSRNNRecurrentImageState? = mps_borrow(recurrentInputStateHandle)
    let outputStates = NSMutableArray()
    layer.encodeSequence(
        commandBuffer: commandBuffer,
        sourceImages: sourceImages,
        destinationImages: destinationImages,
        recurrentInputState: recurrentInputState,
        recurrentOutputStates: outputStates
    )
    guard let lastState = outputStates.lastObject as? MPSRNNRecurrentImageState else { return nil }
    return mps_retain(lastState)
}
