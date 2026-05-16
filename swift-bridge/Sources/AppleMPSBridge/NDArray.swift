import Metal
import MetalPerformanceShaders

private func mps_with_ndarray_dimension_sizes<T>(
    count: Int,
    rawSizes: UnsafePointer<UInt>,
    _ body: (UnsafeMutablePointer<Int>) -> T
) -> T? {
    guard count > 0 else { return nil }
    var sizes = UnsafeBufferPointer(start: rawSizes, count: count).map { Int($0) }
    return sizes.withUnsafeMutableBufferPointer { buffer in
        guard let baseAddress = buffer.baseAddress else { return nil }
        return body(baseAddress)
    }
}

@_cdecl("mps_ndarray_descriptor_new_with_dimension_sizes")
public func mps_ndarray_descriptor_new_with_dimension_sizes(
    _ dataTypeRaw: UInt32,
    _ numberOfDimensions: Int,
    _ dimensionSizes: UnsafePointer<UInt>?
) -> UnsafeMutableRawPointer? {
    guard let dataType = mps_data_type(dataTypeRaw),
          let dimensionSizes
    else {
        return nil
    }
    guard let descriptor = mps_with_ndarray_dimension_sizes(
        count: numberOfDimensions,
        rawSizes: dimensionSizes,
        {
            MPSNDArrayDescriptor(
                dataType: dataType,
                dimensionCount: numberOfDimensions,
                dimensionSizes: $0
            )
        }
    ) else {
        return nil
    }
    return mps_retain(descriptor)
}

@_cdecl("mps_ndarray_descriptor_data_type")
public func mps_ndarray_descriptor_data_type(_ handle: UnsafeMutableRawPointer?) -> UInt32 {
    guard let descriptor: MPSNDArrayDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.dataType.rawValue
}

@_cdecl("mps_ndarray_descriptor_set_data_type")
public func mps_ndarray_descriptor_set_data_type(
    _ handle: UnsafeMutableRawPointer?,
    _ dataTypeRaw: UInt32
) {
    guard let descriptor: MPSNDArrayDescriptor = mps_borrow(handle),
          let dataType = mps_data_type(dataTypeRaw)
    else {
        return
    }
    descriptor.dataType = dataType
}

@_cdecl("mps_ndarray_descriptor_number_of_dimensions")
public func mps_ndarray_descriptor_number_of_dimensions(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let descriptor: MPSNDArrayDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.numberOfDimensions
}

@_cdecl("mps_ndarray_descriptor_set_number_of_dimensions")
public func mps_ndarray_descriptor_set_number_of_dimensions(
    _ handle: UnsafeMutableRawPointer?,
    _ numberOfDimensions: Int
) {
    guard let descriptor: MPSNDArrayDescriptor = mps_borrow(handle) else { return }
    descriptor.numberOfDimensions = numberOfDimensions
}

@_cdecl("mps_ndarray_descriptor_length_of_dimension")
public func mps_ndarray_descriptor_length_of_dimension(
    _ handle: UnsafeMutableRawPointer?,
    _ dimensionIndex: Int
) -> Int {
    guard let descriptor: MPSNDArrayDescriptor = mps_borrow(handle) else { return 0 }
    return descriptor.length(ofDimension: dimensionIndex)
}

@_cdecl("mps_ndarray_descriptor_reshape_with_dimension_sizes")
public func mps_ndarray_descriptor_reshape_with_dimension_sizes(
    _ handle: UnsafeMutableRawPointer?,
    _ numberOfDimensions: Int,
    _ dimensionSizes: UnsafePointer<UInt>?
) {
    guard let descriptor: MPSNDArrayDescriptor = mps_borrow(handle),
          let dimensionSizes
    else {
        return
    }
    descriptor.numberOfDimensions = numberOfDimensions
    _ = mps_with_ndarray_dimension_sizes(
        count: numberOfDimensions,
        rawSizes: dimensionSizes,
        {
            descriptor.reshape(
                withDimensionCount: numberOfDimensions,
                dimensionSizes: $0
            )
        }
    )
}

@_cdecl("mps_ndarray_descriptor_transpose_dimension")
public func mps_ndarray_descriptor_transpose_dimension(
    _ handle: UnsafeMutableRawPointer?,
    _ dimensionIndex: Int,
    _ otherDimensionIndex: Int
) {
    guard let descriptor: MPSNDArrayDescriptor = mps_borrow(handle) else { return }
    descriptor.transposeDimension(dimensionIndex, withDimension: otherDimensionIndex)
}

@_cdecl("mps_ndarray_new_with_descriptor")
public func mps_ndarray_new_with_descriptor(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ descriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let descriptor: MPSNDArrayDescriptor = mps_borrow(descriptorHandle)
    else {
        return nil
    }
    return mps_retain(MPSNDArray(device: device, descriptor: descriptor))
}

@_cdecl("mps_ndarray_new_scalar")
public func mps_ndarray_new_scalar(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ value: Double
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSNDArray(device: device, scalar: value))
}

@_cdecl("mps_ndarray_new_with_buffer")
public func mps_ndarray_new_with_buffer(
    _ bufferHandle: UnsafeMutableRawPointer?,
    _ offset: Int,
    _ descriptorHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let buffer: MTLBuffer = mps_borrow(bufferHandle),
          let descriptor: MPSNDArrayDescriptor = mps_borrow(descriptorHandle)
    else {
        return nil
    }
    if #available(macOS 15.0, *) {
        return mps_retain(MPSNDArray(buffer: buffer, offset: offset, descriptor: descriptor))
    }
    return nil
}

@_cdecl("mps_ndarray_data_type")
public func mps_ndarray_data_type(_ handle: UnsafeMutableRawPointer?) -> UInt32 {
    guard let array: MPSNDArray = mps_borrow(handle) else { return 0 }
    return array.dataType.rawValue
}

@_cdecl("mps_ndarray_number_of_dimensions")
public func mps_ndarray_number_of_dimensions(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let array: MPSNDArray = mps_borrow(handle) else { return 0 }
    return array.numberOfDimensions
}

@_cdecl("mps_ndarray_length_of_dimension")
public func mps_ndarray_length_of_dimension(
    _ handle: UnsafeMutableRawPointer?,
    _ dimensionIndex: Int
) -> Int {
    guard let array: MPSNDArray = mps_borrow(handle) else { return 0 }
    return array.length(ofDimension: dimensionIndex)
}

@_cdecl("mps_ndarray_descriptor")
public func mps_ndarray_descriptor(_ handle: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let array: MPSNDArray = mps_borrow(handle) else { return nil }
    return mps_retain(array.descriptor())
}

@_cdecl("mps_ndarray_resource_size")
public func mps_ndarray_resource_size(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let array: MPSNDArray = mps_borrow(handle) else { return 0 }
    return array.resourceSize()
}

@_cdecl("mps_ndarray_identity_new")
public func mps_ndarray_identity_new(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    if #available(macOS 15.0, *) {
        return mps_retain(MPSNDArrayIdentity(device: device))
    }
    return nil
}

@_cdecl("mps_ndarray_identity_reshape")
public func mps_ndarray_identity_reshape(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ sourceArrayHandle: UnsafeMutableRawPointer?,
    _ numberOfDimensions: Int,
    _ dimensionSizes: UnsafePointer<UInt>?,
    _ destinationArrayHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard #available(macOS 15.0, *) else { return nil }
    guard let identity: MPSNDArrayIdentity = mps_borrow(handle),
          let sourceArray: MPSNDArray = mps_borrow(sourceArrayHandle),
          let dimensionSizes
    else {
        return nil
    }

    let commandBuffer: MTLCommandBuffer? = mps_borrow(commandBufferHandle)
    let destinationArray: MPSNDArray? = mps_borrow(destinationArrayHandle)
    let shape = UnsafeBufferPointer(start: dimensionSizes, count: numberOfDimensions).map {
        NSNumber(value: $0)
    }
    guard let result = identity.reshape(
        with: nil,
        commandBuffer: commandBuffer,
        sourceArray: sourceArray,
        shape: shape,
        destinationArray: destinationArray
    ) else {
        return nil
    }
    return mps_retain(result)
}
