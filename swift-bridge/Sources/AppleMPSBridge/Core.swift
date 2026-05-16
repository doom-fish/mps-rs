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
