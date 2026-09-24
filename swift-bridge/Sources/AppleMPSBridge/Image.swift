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
          (1...4).contains(channelFormatRaw),
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
    guard let texture: MTLTexture = mps_borrow(textureHandle),
          mps_texture_matches_feature_channels(texture, featureChannels)
    else {
        return nil
    }
    return mps_retain(MPSImage(texture: texture, featureChannels: featureChannels))
}

private func mps_texture_channel_count(_ format: MTLPixelFormat) -> Int? {
    switch format {
    case .r8Unorm, .r8Snorm, .r16Unorm, .r16Snorm, .r16Float, .r32Float:
        return 1
    case .rg8Unorm, .rg8Snorm, .rg16Unorm, .rg16Snorm, .rg16Float, .rg32Float:
        return 2
    case .rgba8Unorm, .rgba16Unorm, .rgba16Float, .rgba32Float, .bgra8Unorm:
        return 4
    default:
        return nil
    }
}

private func mps_texture_matches_feature_channels(_ texture: MTLTexture, _ featureChannels: Int) -> Bool {
    guard featureChannels > 0,
          let channels = mps_texture_channel_count(texture.pixelFormat)
    else {
        return false
    }
    switch texture.textureType {
    case .type2D:
        return channels == 4 ? featureChannels == 3 || featureChannels == 4 : featureChannels == channels
    case .type2DArray:
        if featureChannels <= 4 {
            return channels == 4 ? featureChannels >= 3 : featureChannels == channels
        }
        let slices = (featureChannels + 3) / 4
        return channels == 4 && texture.arrayLength % slices == 0
    default:
        return false
    }
}

private func mps_feature_channel_element_size(_ format: MPSImageFeatureChannelFormat) -> Int? {
    switch format {
    case .unorm8:
        return 1
    case .unorm16, .float16:
        return 2
    case .float32:
        return 4
    default:
        return nil
    }
}

private func mps_image_transfer_is_valid(
    _ image: MPSImage,
    _ dataLength: Int,
    _ layout: MPSDataLayout,
    _ bytesPerRow: Int,
    _ region: MTLRegion,
    _ featureChannelOffset: Int,
    _ featureChannelCount: Int,
    _ imageIndex: Int
) -> Bool {
    guard !(image is MPSTemporaryImage),
          layout == .HeightxWidthxFeatureChannels || layout == .featureChannelsxHeightxWidth,
          let elementSize = mps_feature_channel_element_size(image.featureChannelFormat),
          region.size.width > 0, region.size.height > 0,
          region.origin.z == 0, region.size.depth == 1,
          region.origin.x >= 0, region.origin.y >= 0,
          region.origin.x <= image.width - region.size.width,
          region.origin.y <= image.height - region.size.height,
          imageIndex >= 0, imageIndex < image.numberOfImages,
          featureChannelCount > 0, featureChannelOffset >= 0,
          featureChannelOffset % 4 == 0,
          featureChannelOffset <= image.featureChannels - featureChannelCount,
          featureChannelCount % 4 == 0
              || featureChannelOffset + featureChannelCount == image.featureChannels
    else {
        return false
    }
    let chunky = layout == .HeightxWidthxFeatureChannels
    let (rowElements, rowOverflow) = region.size.width.multipliedReportingOverflow(
        by: chunky ? featureChannelCount : 1
    )
    let (rowBytes, bytesOverflow) = rowElements.multipliedReportingOverflow(by: elementSize)
    let (planeBytes, planeOverflow) = bytesPerRow.multipliedReportingOverflow(by: region.size.height)
    let (required, requiredOverflow) = planeBytes.multipliedReportingOverflow(
        by: chunky ? 1 : featureChannelCount
    )
    guard !rowOverflow, !bytesOverflow, !planeOverflow, !requiredOverflow,
          bytesPerRow >= rowBytes, dataLength >= required
    else {
        return false
    }
    let storageMode = image.texture.storageMode
    return storageMode == .shared || storageMode == .managed
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

@_cdecl("mps_image_feature_channel_format")
public func mps_image_feature_channel_format(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let image: MPSImage = mps_borrow(handle) else { return 0 }
    return image.featureChannelFormat.rawValue
}

@_cdecl("mps_image_pixel_format")
public func mps_image_pixel_format(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let image: MPSImage = mps_borrow(handle) else { return 0 }
    return image.pixelFormat.rawValue
}

@_cdecl("mps_image_texture_info")
public func mps_image_texture_info(
    _ handle: UnsafeMutableRawPointer?,
    _ texture: UnsafeMutablePointer<UnsafeMutableRawPointer?>?,
    _ textureType: UnsafeMutablePointer<UInt>?,
    _ arrayLength: UnsafeMutablePointer<Int>?,
    _ usage: UnsafeMutablePointer<UInt>?
) -> Bool {
    guard let image: MPSImage = mps_borrow(handle) else { return false }
    let backing = image.texture
    texture?.pointee = Unmanaged.passUnretained(backing as AnyObject).toOpaque()
    textureType?.pointee = backing.textureType.rawValue
    arrayLength?.pointee = backing.arrayLength
    usage?.pointee = backing.usage.rawValue
    return true
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
    _ dataLength: Int,
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
    let region = mps_region(x, y, z, width, height, depth)
    guard let image: MPSImage = mps_borrow(handle),
          let data,
          let layout = mps_data_layout(dataLayoutRaw),
          mps_image_transfer_is_valid(
              image,
              dataLength,
              layout,
              bytesPerRow,
              region,
              featureChannelOffset,
              featureChannelCount,
              imageIndex
          )
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
        region: region,
        featureChannelInfo: params,
        imageIndex: imageIndex
    )
    return true
}

@_cdecl("mps_image_write_bytes")
public func mps_image_write_bytes(
    _ handle: UnsafeMutableRawPointer?,
    _ data: UnsafeRawPointer?,
    _ dataLength: Int,
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
    let region = mps_region(x, y, z, width, height, depth)
    guard let image: MPSImage = mps_borrow(handle),
          let data,
          let layout = mps_data_layout(dataLayoutRaw),
          mps_image_transfer_is_valid(
              image,
              dataLength,
              layout,
              bytesPerRow,
              region,
              featureChannelOffset,
              featureChannelCount,
              imageIndex
          )
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
        region: region,
        featureChannelInfo: params,
        imageIndex: imageIndex
    )
    return true
}
