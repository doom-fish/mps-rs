import Metal
import MetalPerformanceShaders
import simd

@inline(__always)
func mps_unary_kernel(_ handle: UnsafeMutableRawPointer?) -> MPSUnaryImageKernel? {
    mps_borrow(handle)
}

@inline(__always)
func mps_binary_kernel(_ handle: UnsafeMutableRawPointer?) -> MPSBinaryImageKernel? {
    mps_borrow(handle)
}

@_cdecl("mps_unary_encode_image")
public func mps_unary_encode_image(
    _ kernelHandle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ destinationHandle: UnsafeMutableRawPointer?
) {
    guard let kernel = mps_unary_kernel(kernelHandle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let source: MPSImage = mps_borrow(sourceHandle),
          let destination: MPSImage = mps_borrow(destinationHandle)
    else {
        return
    }

    kernel.encode(commandBuffer: commandBuffer, sourceImage: source, destinationImage: destination)
}

@_cdecl("mps_unary_encode_texture")
public func mps_unary_encode_texture(
    _ kernelHandle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ sourceTextureHandle: UnsafeMutableRawPointer?,
    _ destinationTextureHandle: UnsafeMutableRawPointer?
) {
    guard let kernel = mps_unary_kernel(kernelHandle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let sourceTexture: MTLTexture = mps_borrow(sourceTextureHandle),
          let destinationTexture: MTLTexture = mps_borrow(destinationTextureHandle)
    else {
        return
    }

    kernel.encode(commandBuffer: commandBuffer, sourceTexture: sourceTexture, destinationTexture: destinationTexture)
}

@_cdecl("mps_unary_set_edge_mode")
public func mps_unary_set_edge_mode(_ kernelHandle: UnsafeMutableRawPointer?, _ edgeModeRaw: UInt) {
    guard let kernel = mps_unary_kernel(kernelHandle) else { return }
    kernel.edgeMode = mps_image_edge_mode(edgeModeRaw)
}

@_cdecl("mps_unary_set_clip_rect")
public func mps_unary_set_clip_rect(
    _ kernelHandle: UnsafeMutableRawPointer?,
    _ x: Int,
    _ y: Int,
    _ z: Int,
    _ width: Int,
    _ height: Int,
    _ depth: Int
) {
    guard let kernel = mps_unary_kernel(kernelHandle) else { return }
    kernel.clipRect = mps_region(x, y, z, width, height, depth)
}

@_cdecl("mps_image_scale_set_transform")
public func mps_image_scale_set_transform(
    _ kernelHandle: UnsafeMutableRawPointer?,
    _ scaleX: Double,
    _ scaleY: Double,
    _ translateX: Double,
    _ translateY: Double
) {
    guard let kernel: MPSImageScale = mps_borrow(kernelHandle) else { return }
    var transform = MPSScaleTransform(
        scaleX: scaleX,
        scaleY: scaleY,
        translateX: translateX,
        translateY: translateY
    )
    withUnsafePointer(to: &transform) { pointer in
        kernel.scaleTransform = pointer
    }
}

@_cdecl("mps_binary_encode_image")
public func mps_binary_encode_image(
    _ kernelHandle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ primaryHandle: UnsafeMutableRawPointer?,
    _ secondaryHandle: UnsafeMutableRawPointer?,
    _ destinationHandle: UnsafeMutableRawPointer?
) {
    guard let kernel = mps_binary_kernel(kernelHandle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let primary: MPSImage = mps_borrow(primaryHandle),
          let secondary: MPSImage = mps_borrow(secondaryHandle),
          let destination: MPSImage = mps_borrow(destinationHandle)
    else {
        return
    }

    kernel.encode(commandBuffer: commandBuffer, primaryImage: primary, secondaryImage: secondary, destinationImage: destination)
}

@_cdecl("mps_binary_encode_texture")
public func mps_binary_encode_texture(
    _ kernelHandle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ primaryTextureHandle: UnsafeMutableRawPointer?,
    _ secondaryTextureHandle: UnsafeMutableRawPointer?,
    _ destinationTextureHandle: UnsafeMutableRawPointer?
) {
    guard let kernel = mps_binary_kernel(kernelHandle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let primaryTexture: MTLTexture = mps_borrow(primaryTextureHandle),
          let secondaryTexture: MTLTexture = mps_borrow(secondaryTextureHandle),
          let destinationTexture: MTLTexture = mps_borrow(destinationTextureHandle)
    else {
        return
    }

    kernel.encode(
        commandBuffer: commandBuffer,
        primaryTexture: primaryTexture,
        secondaryTexture: secondaryTexture,
        destinationTexture: destinationTexture
    )
}

@_cdecl("mps_binary_set_primary_edge_mode")
public func mps_binary_set_primary_edge_mode(_ kernelHandle: UnsafeMutableRawPointer?, _ edgeModeRaw: UInt) {
    guard let kernel = mps_binary_kernel(kernelHandle) else { return }
    kernel.primaryEdgeMode = mps_image_edge_mode(edgeModeRaw)
}

@_cdecl("mps_binary_set_secondary_edge_mode")
public func mps_binary_set_secondary_edge_mode(_ kernelHandle: UnsafeMutableRawPointer?, _ edgeModeRaw: UInt) {
    guard let kernel = mps_binary_kernel(kernelHandle) else { return }
    kernel.secondaryEdgeMode = mps_image_edge_mode(edgeModeRaw)
}

@_cdecl("mps_binary_set_clip_rect")
public func mps_binary_set_clip_rect(
    _ kernelHandle: UnsafeMutableRawPointer?,
    _ x: Int,
    _ y: Int,
    _ z: Int,
    _ width: Int,
    _ height: Int,
    _ depth: Int
) {
    guard let kernel = mps_binary_kernel(kernelHandle) else { return }
    kernel.clipRect = mps_region(x, y, z, width, height, depth)
}

@_cdecl("mps_image_arithmetic_set_scales_bias")
public func mps_image_arithmetic_set_scales_bias(
    _ kernelHandle: UnsafeMutableRawPointer?,
    _ primaryScale: Float,
    _ secondaryScale: Float,
    _ bias: Float
) {
    guard let kernel: MPSImageArithmetic = mps_borrow(kernelHandle) else { return }
    kernel.primaryScale = primaryScale
    kernel.secondaryScale = secondaryScale
    kernel.bias = bias
}

@_cdecl("mps_image_arithmetic_set_clamp")
public func mps_image_arithmetic_set_clamp(
    _ kernelHandle: UnsafeMutableRawPointer?,
    _ minimumValue: Float,
    _ maximumValue: Float
) {
    guard let kernel: MPSImageArithmetic = mps_borrow(kernelHandle) else { return }
    kernel.minimumValue = minimumValue
    kernel.maximumValue = maximumValue
}

@_cdecl("mps_image_gaussian_blur_new")
public func mps_image_gaussian_blur_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ sigma: Float
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSImageGaussianBlur(device: device, sigma: sigma))
}

@_cdecl("mps_image_box_new")
public func mps_image_box_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ kernelWidth: Int,
    _ kernelHeight: Int
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSImageBox(device: device, kernelWidth: kernelWidth, kernelHeight: kernelHeight))
}

@_cdecl("mps_image_sobel_new")
public func mps_image_sobel_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ transform: UnsafePointer<Float>?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    if let transform {
        return mps_retain(MPSImageSobel(device: device, linearGrayColorTransform: transform))
    }
    return mps_retain(MPSImageSobel(device: device))
}

@_cdecl("mps_image_median_new")
public func mps_image_median_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ kernelDiameter: Int
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSImageMedian(device: device, kernelDiameter: kernelDiameter))
}

@_cdecl("mps_image_convolution_new")
public func mps_image_convolution_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ kernelWidth: Int,
    _ kernelHeight: Int,
    _ weights: UnsafePointer<Float>?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let weights
    else {
        return nil
    }
    return mps_retain(
        MPSImageConvolution(
            device: device,
            kernelWidth: kernelWidth,
            kernelHeight: kernelHeight,
            weights: weights
        )
    )
}

@_cdecl("mps_image_bilinear_scale_new")
public func mps_image_bilinear_scale_new(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSImageBilinearScale(device: device))
}

@_cdecl("mps_image_lanczos_scale_new")
public func mps_image_lanczos_scale_new(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSImageLanczosScale(device: device))
}

@_cdecl("mps_image_threshold_binary_new")
public func mps_image_threshold_binary_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ thresholdValue: Float,
    _ maximumValue: Float,
    _ transform: UnsafePointer<Float>?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(
        MPSImageThresholdBinary(
            device: device,
            thresholdValue: thresholdValue,
            maximumValue: maximumValue,
            linearGrayColorTransform: transform
        )
    )
}

@_cdecl("mps_image_histogram_new")
public func mps_image_histogram_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ numberOfEntries: Int,
    _ histogramForAlpha: Bool,
    _ minValues: UnsafePointer<Float>?,
    _ maxValues: UnsafePointer<Float>?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let minValues,
          let maxValues
    else {
        return nil
    }

    var info = MPSImageHistogramInfo()
    info.numberOfHistogramEntries = numberOfEntries
    info.histogramForAlpha = ObjCBool(histogramForAlpha)
    info.minPixelValue = vector_float4(minValues[0], minValues[1], minValues[2], minValues[3])
    info.maxPixelValue = vector_float4(maxValues[0], maxValues[1], maxValues[2], maxValues[3])
    return withUnsafePointer(to: &info) { pointer in
        mps_retain(MPSImageHistogram(device: device, histogramInfo: pointer))
    }
}

@_cdecl("mps_image_histogram_encode_image")
public func mps_image_histogram_encode_image(
    _ histogramHandle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ sourceHandle: UnsafeMutableRawPointer?,
    _ histogramBufferHandle: UnsafeMutableRawPointer?,
    _ histogramOffset: Int
) {
    guard let histogram: MPSImageHistogram = mps_borrow(histogramHandle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let source: MPSImage = mps_borrow(sourceHandle),
          let histogramBuffer: MTLBuffer = mps_borrow(histogramBufferHandle)
    else {
        return
    }

    histogram.encode(
        to: commandBuffer,
        sourceTexture: source.texture,
        histogram: histogramBuffer,
        histogramOffset: histogramOffset
    )
}

@_cdecl("mps_image_histogram_encode_texture")
public func mps_image_histogram_encode_texture(
    _ histogramHandle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ sourceTextureHandle: UnsafeMutableRawPointer?,
    _ histogramBufferHandle: UnsafeMutableRawPointer?,
    _ histogramOffset: Int
) {
    guard let histogram: MPSImageHistogram = mps_borrow(histogramHandle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let sourceTexture: MTLTexture = mps_borrow(sourceTextureHandle),
          let histogramBuffer: MTLBuffer = mps_borrow(histogramBufferHandle)
    else {
        return
    }

    histogram.encode(
        to: commandBuffer,
        sourceTexture: sourceTexture,
        histogram: histogramBuffer,
        histogramOffset: histogramOffset
    )
}

@_cdecl("mps_image_histogram_size_for_source_format")
public func mps_image_histogram_size_for_source_format(
    _ histogramHandle: UnsafeMutableRawPointer?,
    _ sourceFormatRaw: UInt
) -> Int {
    guard let histogram: MPSImageHistogram = mps_borrow(histogramHandle),
          let sourceFormat = MTLPixelFormat(rawValue: sourceFormatRaw)
    else {
        return 0
    }
    return histogram.histogramSize(forSourceFormat: sourceFormat)
}

@_cdecl("mps_image_statistics_min_max_new")
public func mps_image_statistics_min_max_new(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSImageStatisticsMinAndMax(device: device))
}

@_cdecl("mps_image_statistics_mean_new")
public func mps_image_statistics_mean_new(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSImageStatisticsMean(device: device))
}

@_cdecl("mps_image_reduce_row_min_new")
public func mps_image_reduce_row_min_new(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSImageReduceRowMin(device: device))
}

@_cdecl("mps_image_reduce_row_max_new")
public func mps_image_reduce_row_max_new(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSImageReduceRowMax(device: device))
}

@_cdecl("mps_image_reduce_row_mean_new")
public func mps_image_reduce_row_mean_new(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSImageReduceRowMean(device: device))
}

@_cdecl("mps_image_reduce_row_sum_new")
public func mps_image_reduce_row_sum_new(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSImageReduceRowSum(device: device))
}

@_cdecl("mps_image_add_new")
public func mps_image_add_new(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSImageAdd(device: device))
}
