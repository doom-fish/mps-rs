import Metal
import MetalPerformanceShaders

@_cdecl("mps_image_new_with_descriptor")
public func mps_image_new_with_descriptor(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ channelFormatRaw: UInt,
    _ width: Int,
    _ height: Int,
    _ featureChannels: Int,
    _ numberOfImages: Int,
    _ usageRaw: UInt,
    _ storageModeRaw: UInt
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let channelFormat = mps_channel_format(channelFormatRaw)
    else {
        return nil
    }

    let descriptor = MPSImageDescriptor(
        channelFormat: channelFormat,
        width: width,
        height: height,
        featureChannels: featureChannels,
        numberOfImages: numberOfImages,
        usage: MTLTextureUsage(rawValue: usageRaw)
    )
    descriptor.storageMode = mps_storage_mode(storageModeRaw)
    return mps_retain(MPSImage(device: device, imageDescriptor: descriptor))
}

@_cdecl("mps_image_new_with_texture")
public func mps_image_new_with_texture(
    _ textureHandle: UnsafeMutableRawPointer?,
    _ featureChannels: Int
) -> UnsafeMutableRawPointer? {
    guard let texture: MTLTexture = mps_borrow(textureHandle) else { return nil }
    return mps_retain(MPSImage(texture: texture, featureChannels: featureChannels))
}

@_cdecl("mps_image_width")
public func mps_image_width(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let image: MPSImage = mps_borrow(handle) else { return 0 }
    return image.width
}

@_cdecl("mps_image_height")
public func mps_image_height(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let image: MPSImage = mps_borrow(handle) else { return 0 }
    return image.height
}

@_cdecl("mps_image_feature_channels")
public func mps_image_feature_channels(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let image: MPSImage = mps_borrow(handle) else { return 0 }
    return image.featureChannels
}

@_cdecl("mps_image_number_of_images")
public func mps_image_number_of_images(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let image: MPSImage = mps_borrow(handle) else { return 0 }
    return image.numberOfImages
}

@_cdecl("mps_image_pixel_size")
public func mps_image_pixel_size(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let image: MPSImage = mps_borrow(handle) else { return 0 }
    return image.pixelSize
}

@_cdecl("mps_image_pixel_format")
public func mps_image_pixel_format(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let image: MPSImage = mps_borrow(handle) else { return 0 }
    return image.pixelFormat.rawValue
}

@_cdecl("mps_get_image_type")
public func mps_get_image_type(_ handle: UnsafeMutableRawPointer?) -> UInt32 {
    guard let image: MPSImage = mps_borrow(handle) else { return 0 }
    return MPSGetImageType(image).rawValue
}

@_cdecl("mps_image_batch_increment_read_count")
public func mps_image_batch_increment_read_count(
    _ handles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ count: Int,
    _ amount: Int
) -> Int {
    guard let images: [MPSImage] = mps_borrow_array(handles, count: count) else { return 0 }
    if images.isEmpty { return 0 }
    return MPSImageBatchIncrementReadCount(images, amount)
}

@_cdecl("mps_image_batch_synchronize")
public func mps_image_batch_synchronize(
    _ handles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ count: Int,
    _ commandBufferHandle: UnsafeMutableRawPointer?
) {
    guard let images: [MPSImage] = mps_borrow_array(handles, count: count),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle)
    else {
        return
    }
    if images.isEmpty { return }
    MPSImageBatchSynchronize(images, commandBuffer)
}

@_cdecl("mps_image_batch_resource_size")
public func mps_image_batch_resource_size(
    _ handles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ count: Int
) -> Int {
    guard let images: [MPSImage] = mps_borrow_array(handles, count: count) else { return 0 }
    if images.isEmpty { return 0 }
    return MPSImageBatchResourceSize(images)
}

@_cdecl("mps_image_read_bytes")
public func mps_image_read_bytes(
    _ handle: UnsafeMutableRawPointer?,
    _ data: UnsafeMutableRawPointer?,
    _ dataLayoutRaw: UInt,
    _ bytesPerRow: Int,
    _ x: Int,
    _ y: Int,
    _ z: Int,
    _ width: Int,
    _ height: Int,
    _ depth: Int,
    _ featureChannelOffset: Int,
    _ featureChannelCount: Int,
    _ imageIndex: Int
) -> Bool {
    guard let image: MPSImage = mps_borrow(handle),
          let data,
          let layout = mps_data_layout(dataLayoutRaw)
    else {
        return false
    }

    let params = MPSImageReadWriteParams(
        featureChannelOffset: featureChannelOffset,
        numberOfFeatureChannelsToReadWrite: featureChannelCount
    )
    image.readBytes(
        data,
        dataLayout: layout,
        bytesPerRow: bytesPerRow,
        region: mps_region(x, y, z, width, height, depth),
        featureChannelInfo: params,
        imageIndex: imageIndex
    )
    return true
}

@_cdecl("mps_image_write_bytes")
public func mps_image_write_bytes(
    _ handle: UnsafeMutableRawPointer?,
    _ data: UnsafeRawPointer?,
    _ dataLayoutRaw: UInt,
    _ bytesPerRow: Int,
    _ x: Int,
    _ y: Int,
    _ z: Int,
    _ width: Int,
    _ height: Int,
    _ depth: Int,
    _ featureChannelOffset: Int,
    _ featureChannelCount: Int,
    _ imageIndex: Int
) -> Bool {
    guard let image: MPSImage = mps_borrow(handle),
          let data,
          let layout = mps_data_layout(dataLayoutRaw)
    else {
        return false
    }

    let params = MPSImageReadWriteParams(
        featureChannelOffset: featureChannelOffset,
        numberOfFeatureChannelsToReadWrite: featureChannelCount
    )
    image.writeBytes(
        data,
        dataLayout: layout,
        bytesPerRow: bytesPerRow,
        region: mps_region(x, y, z, width, height, depth),
        featureChannelInfo: params,
        imageIndex: imageIndex
    )
    return true
}
