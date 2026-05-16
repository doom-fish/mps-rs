import Foundation
import Metal
import MetalPerformanceShaders

@inline(__always)
public func mps_retain<T: AnyObject>(_ object: T) -> UnsafeMutableRawPointer {
    Unmanaged.passRetained(object as AnyObject).toOpaque()
}

@inline(__always)
public func mps_release(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    Unmanaged<AnyObject>.fromOpaque(handle).release()
}

@inline(__always)
public func mps_borrow<T>(_ handle: UnsafeMutableRawPointer?) -> T? {
    guard let handle else { return nil }
    return Unmanaged<AnyObject>.fromOpaque(handle).takeUnretainedValue() as? T
}

@_cdecl("mps_object_release")
public func mps_object_release(_ handle: UnsafeMutableRawPointer?) {
    mps_release(handle)
}

@inline(__always)
func mps_region(
    _ x: Int,
    _ y: Int,
    _ z: Int,
    _ width: Int,
    _ height: Int,
    _ depth: Int
) -> MTLRegion {
    MTLRegion(
        origin: MTLOrigin(x: x, y: y, z: z),
        size: MTLSize(width: width, height: height, depth: depth)
    )
}

@inline(__always)
func mps_channel_format(_ raw: UInt) -> MPSImageFeatureChannelFormat? {
    MPSImageFeatureChannelFormat(rawValue: raw)
}

@inline(__always)
func mps_data_layout(_ raw: UInt) -> MPSDataLayout? {
    MPSDataLayout(rawValue: raw)
}

@inline(__always)
func mps_image_edge_mode(_ raw: UInt) -> MPSImageEdgeMode {
    switch raw {
    case 1: return .clamp
    default: return .zero
    }
}

@inline(__always)
func mps_storage_mode(_ raw: UInt) -> MTLStorageMode {
    switch raw {
    case 0: return .shared
    case 2: return .private
    case 3: return .memoryless
    default: return .managed
    }
}

@inline(__always)
func mps_data_type(_ raw: UInt32) -> MPSDataType? {
    MPSDataType(rawValue: raw)
}

@_cdecl("mps_supports_mtl_device")
public func mps_supports_mtl_device(_ deviceHandle: UnsafeMutableRawPointer?) -> Bool {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return false }
    return MPSSupportsMTLDevice(device)
}

@_cdecl("mps_get_preferred_device")
public func mps_get_preferred_device(_ optionsRaw: UInt) -> UnsafeMutableRawPointer? {
    guard let device = MPSGetPreferredDevice(MPSDeviceOptions(rawValue: optionsRaw)) else {
        return nil
    }
    return mps_retain(device)
}

@_cdecl("mps_hint_temporary_memory_high_water_mark")
public func mps_hint_temporary_memory_high_water_mark(
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ bytes: Int
) {
    guard let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle) else { return }
    MPSHintTemporaryMemoryHighWaterMark(commandBuffer, bytes)
}

@_cdecl("mps_set_heap_cache_duration")
public func mps_set_heap_cache_duration(
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ seconds: Double
) {
    guard let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle) else { return }
    MPSSetHeapCacheDuration(commandBuffer, seconds)
}
