import Metal
import MetalPerformanceShaders

@inline(__always)
private func mps_polygon_type(_ raw: UInt) -> MPSPolygonType? {
    MPSPolygonType(rawValue: raw)
}

@inline(__always)
private func mps_acceleration_structure_usage(_ raw: UInt) -> MPSAccelerationStructureUsage {
    MPSAccelerationStructureUsage(rawValue: raw)
}

@inline(__always)
private func mps_cull_mode(_ raw: UInt) -> MTLCullMode? {
    MTLCullMode(rawValue: raw)
}

@inline(__always)
private func mps_winding(_ raw: UInt) -> MTLWinding? {
    MTLWinding(rawValue: raw)
}

@inline(__always)
private func mps_ray_data_type(_ raw: UInt) -> MPSRayDataType? {
    MPSRayDataType(rawValue: raw)
}

@inline(__always)
private func mps_intersection_data_type(_ raw: UInt) -> MPSIntersectionDataType? {
    MPSIntersectionDataType(rawValue: raw)
}

@inline(__always)
private func mps_intersection_type(_ raw: UInt) -> MPSIntersectionType? {
    MPSIntersectionType(rawValue: raw)
}

@_cdecl("mps_polygon_acceleration_structure_new")
public func mps_polygon_acceleration_structure_new(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSPolygonAccelerationStructure(device: device))
}

@_cdecl("mps_polygon_acceleration_structure_polygon_type")
public func mps_polygon_acceleration_structure_polygon_type(
    _ handle: UnsafeMutableRawPointer?
) -> UInt {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle) else { return 0 }
    return accelerationStructure.polygonType.rawValue
}

@_cdecl("mps_polygon_acceleration_structure_set_polygon_type")
public func mps_polygon_acceleration_structure_set_polygon_type(
    _ handle: UnsafeMutableRawPointer?,
    _ polygonTypeRaw: UInt
) {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle),
          let polygonType = mps_polygon_type(polygonTypeRaw)
    else {
        return
    }
    accelerationStructure.polygonType = polygonType
}

@_cdecl("mps_polygon_acceleration_structure_vertex_stride")
public func mps_polygon_acceleration_structure_vertex_stride(
    _ handle: UnsafeMutableRawPointer?
) -> Int {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle) else { return 0 }
    return accelerationStructure.vertexStride
}

@_cdecl("mps_polygon_acceleration_structure_set_vertex_stride")
public func mps_polygon_acceleration_structure_set_vertex_stride(
    _ handle: UnsafeMutableRawPointer?,
    _ vertexStride: Int
) {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle) else { return }
    accelerationStructure.vertexStride = vertexStride
}

@_cdecl("mps_polygon_acceleration_structure_index_type")
public func mps_polygon_acceleration_structure_index_type(
    _ handle: UnsafeMutableRawPointer?
) -> UInt32 {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle) else { return 0 }
    return accelerationStructure.indexType.rawValue
}

@_cdecl("mps_polygon_acceleration_structure_set_index_type")
public func mps_polygon_acceleration_structure_set_index_type(
    _ handle: UnsafeMutableRawPointer?,
    _ indexTypeRaw: UInt32
) {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle),
          let indexType = mps_data_type(indexTypeRaw)
    else {
        return
    }
    accelerationStructure.indexType = indexType
}

@_cdecl("mps_polygon_acceleration_structure_set_vertex_buffer")
public func mps_polygon_acceleration_structure_set_vertex_buffer(
    _ handle: UnsafeMutableRawPointer?,
    _ bufferHandle: UnsafeMutableRawPointer?
) {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle) else { return }
    let buffer: MTLBuffer? = mps_borrow(bufferHandle)
    accelerationStructure.vertexBuffer = buffer
}

@_cdecl("mps_polygon_acceleration_structure_vertex_buffer_offset")
public func mps_polygon_acceleration_structure_vertex_buffer_offset(
    _ handle: UnsafeMutableRawPointer?
) -> Int {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle) else { return 0 }
    return accelerationStructure.vertexBufferOffset
}

@_cdecl("mps_polygon_acceleration_structure_set_vertex_buffer_offset")
public func mps_polygon_acceleration_structure_set_vertex_buffer_offset(
    _ handle: UnsafeMutableRawPointer?,
    _ offset: Int
) {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle) else { return }
    accelerationStructure.vertexBufferOffset = offset
}

@_cdecl("mps_polygon_acceleration_structure_set_index_buffer")
public func mps_polygon_acceleration_structure_set_index_buffer(
    _ handle: UnsafeMutableRawPointer?,
    _ bufferHandle: UnsafeMutableRawPointer?
) {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle) else { return }
    let buffer: MTLBuffer? = mps_borrow(bufferHandle)
    accelerationStructure.indexBuffer = buffer
}

@_cdecl("mps_polygon_acceleration_structure_index_buffer_offset")
public func mps_polygon_acceleration_structure_index_buffer_offset(
    _ handle: UnsafeMutableRawPointer?
) -> Int {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle) else { return 0 }
    return accelerationStructure.indexBufferOffset
}

@_cdecl("mps_polygon_acceleration_structure_set_index_buffer_offset")
public func mps_polygon_acceleration_structure_set_index_buffer_offset(
    _ handle: UnsafeMutableRawPointer?,
    _ offset: Int
) {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle) else { return }
    accelerationStructure.indexBufferOffset = offset
}

@_cdecl("mps_polygon_acceleration_structure_polygon_count")
public func mps_polygon_acceleration_structure_polygon_count(
    _ handle: UnsafeMutableRawPointer?
) -> Int {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle) else { return 0 }
    return accelerationStructure.polygonCount
}

@_cdecl("mps_polygon_acceleration_structure_set_polygon_count")
public func mps_polygon_acceleration_structure_set_polygon_count(
    _ handle: UnsafeMutableRawPointer?,
    _ polygonCount: Int
) {
    guard let accelerationStructure: MPSPolygonAccelerationStructure = mps_borrow(handle) else { return }
    accelerationStructure.polygonCount = polygonCount
}

@_cdecl("mps_polygon_acceleration_structure_usage")
public func mps_polygon_acceleration_structure_usage(
    _ handle: UnsafeMutableRawPointer?
) -> UInt {
    guard let accelerationStructure: MPSAccelerationStructure = mps_borrow(handle) else { return 0 }
    return accelerationStructure.usage.rawValue
}

@_cdecl("mps_polygon_acceleration_structure_set_usage")
public func mps_polygon_acceleration_structure_set_usage(
    _ handle: UnsafeMutableRawPointer?,
    _ usageRaw: UInt
) {
    guard let accelerationStructure: MPSAccelerationStructure = mps_borrow(handle) else { return }
    accelerationStructure.usage = mps_acceleration_structure_usage(usageRaw)
}

@_cdecl("mps_polygon_acceleration_structure_status")
public func mps_polygon_acceleration_structure_status(
    _ handle: UnsafeMutableRawPointer?
) -> UInt {
    guard let accelerationStructure: MPSAccelerationStructure = mps_borrow(handle) else { return 0 }
    return accelerationStructure.status.rawValue
}

@_cdecl("mps_polygon_acceleration_structure_rebuild")
public func mps_polygon_acceleration_structure_rebuild(
    _ handle: UnsafeMutableRawPointer?
) {
    guard let accelerationStructure: MPSAccelerationStructure = mps_borrow(handle) else { return }
    accelerationStructure.rebuild()
}

@_cdecl("mps_polygon_acceleration_structure_encode_refit")
public func mps_polygon_acceleration_structure_encode_refit(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?
) {
    guard let accelerationStructure: MPSAccelerationStructure = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle)
    else {
        return
    }
    accelerationStructure.encodeRefit(commandBuffer: commandBuffer)
}

@_cdecl("mps_ray_intersector_new")
public func mps_ray_intersector_new(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSRayIntersector(device: device))
}

@_cdecl("mps_ray_intersector_cull_mode")
public func mps_ray_intersector_cull_mode(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let intersector: MPSRayIntersector = mps_borrow(handle) else { return 0 }
    return intersector.cullMode.rawValue
}

@_cdecl("mps_ray_intersector_set_cull_mode")
public func mps_ray_intersector_set_cull_mode(
    _ handle: UnsafeMutableRawPointer?,
    _ cullModeRaw: UInt
) {
    guard let intersector: MPSRayIntersector = mps_borrow(handle),
          let cullMode = mps_cull_mode(cullModeRaw)
    else {
        return
    }
    intersector.cullMode = cullMode
}

@_cdecl("mps_ray_intersector_front_facing_winding")
public func mps_ray_intersector_front_facing_winding(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let intersector: MPSRayIntersector = mps_borrow(handle) else { return 0 }
    return intersector.frontFacingWinding.rawValue
}

@_cdecl("mps_ray_intersector_set_front_facing_winding")
public func mps_ray_intersector_set_front_facing_winding(
    _ handle: UnsafeMutableRawPointer?,
    _ windingRaw: UInt
) {
    guard let intersector: MPSRayIntersector = mps_borrow(handle),
          let frontFacingWinding = mps_winding(windingRaw)
    else {
        return
    }
    intersector.frontFacingWinding = frontFacingWinding
}

@_cdecl("mps_ray_intersector_ray_stride")
public func mps_ray_intersector_ray_stride(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let intersector: MPSRayIntersector = mps_borrow(handle) else { return 0 }
    return intersector.rayStride
}

@_cdecl("mps_ray_intersector_set_ray_stride")
public func mps_ray_intersector_set_ray_stride(
    _ handle: UnsafeMutableRawPointer?,
    _ stride: Int
) {
    guard let intersector: MPSRayIntersector = mps_borrow(handle) else { return }
    intersector.rayStride = stride
}

@_cdecl("mps_ray_intersector_intersection_stride")
public func mps_ray_intersector_intersection_stride(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let intersector: MPSRayIntersector = mps_borrow(handle) else { return 0 }
    return intersector.intersectionStride
}

@_cdecl("mps_ray_intersector_set_intersection_stride")
public func mps_ray_intersector_set_intersection_stride(
    _ handle: UnsafeMutableRawPointer?,
    _ stride: Int
) {
    guard let intersector: MPSRayIntersector = mps_borrow(handle) else { return }
    intersector.intersectionStride = stride
}

@_cdecl("mps_ray_intersector_ray_data_type")
public func mps_ray_intersector_ray_data_type(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let intersector: MPSRayIntersector = mps_borrow(handle) else { return 0 }
    return intersector.rayDataType.rawValue
}

@_cdecl("mps_ray_intersector_set_ray_data_type")
public func mps_ray_intersector_set_ray_data_type(
    _ handle: UnsafeMutableRawPointer?,
    _ dataTypeRaw: UInt
) {
    guard let intersector: MPSRayIntersector = mps_borrow(handle),
          let dataType = mps_ray_data_type(dataTypeRaw)
    else {
        return
    }
    intersector.rayDataType = dataType
}

@_cdecl("mps_ray_intersector_intersection_data_type")
public func mps_ray_intersector_intersection_data_type(_ handle: UnsafeMutableRawPointer?) -> UInt {
    guard let intersector: MPSRayIntersector = mps_borrow(handle) else { return 0 }
    return intersector.intersectionDataType.rawValue
}

@_cdecl("mps_ray_intersector_set_intersection_data_type")
public func mps_ray_intersector_set_intersection_data_type(
    _ handle: UnsafeMutableRawPointer?,
    _ dataTypeRaw: UInt
) {
    guard let intersector: MPSRayIntersector = mps_borrow(handle),
          let dataType = mps_intersection_data_type(dataTypeRaw)
    else {
        return
    }
    intersector.intersectionDataType = dataType
}

@_cdecl("mps_ray_intersector_recommended_minimum_ray_batch_size")
public func mps_ray_intersector_recommended_minimum_ray_batch_size(
    _ handle: UnsafeMutableRawPointer?,
    _ rayCount: Int
) -> Int {
    guard let intersector: MPSRayIntersector = mps_borrow(handle) else { return 0 }
    return intersector.recommendedMinimumRayBatchSize(rayCount: rayCount)
}

@_cdecl("mps_ray_intersector_encode_intersection")
public func mps_ray_intersector_encode_intersection(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ intersectionTypeRaw: UInt,
    _ rayBufferHandle: UnsafeMutableRawPointer?,
    _ rayBufferOffset: Int,
    _ intersectionBufferHandle: UnsafeMutableRawPointer?,
    _ intersectionBufferOffset: Int,
    _ rayCount: Int,
    _ accelerationStructureHandle: UnsafeMutableRawPointer?
) {
    guard let intersector: MPSRayIntersector = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let intersectionType = mps_intersection_type(intersectionTypeRaw),
          let rayBuffer: MTLBuffer = mps_borrow(rayBufferHandle),
          let intersectionBuffer: MTLBuffer = mps_borrow(intersectionBufferHandle),
          let accelerationStructure: MPSAccelerationStructure = mps_borrow(accelerationStructureHandle)
    else {
        return
    }

    intersector.encodeIntersection(
        commandBuffer: commandBuffer,
        intersectionType: intersectionType,
        rayBuffer: rayBuffer,
        rayBufferOffset: rayBufferOffset,
        intersectionBuffer: intersectionBuffer,
        intersectionBufferOffset: intersectionBufferOffset,
        rayCount: rayCount,
        accelerationStructure: accelerationStructure
    )
}

@_cdecl("mps_svgf_new")
public func mps_svgf_new(_ deviceHandle: UnsafeMutableRawPointer?) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSSVGF(device: device))
}

@_cdecl("mps_svgf_depth_weight")
public func mps_svgf_depth_weight(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let svgf: MPSSVGF = mps_borrow(handle) else { return 0 }
    return svgf.depthWeight
}

@_cdecl("mps_svgf_set_depth_weight")
public func mps_svgf_set_depth_weight(_ handle: UnsafeMutableRawPointer?, _ value: Float) {
    guard let svgf: MPSSVGF = mps_borrow(handle) else { return }
    svgf.depthWeight = value
}

@_cdecl("mps_svgf_normal_weight")
public func mps_svgf_normal_weight(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let svgf: MPSSVGF = mps_borrow(handle) else { return 0 }
    return svgf.normalWeight
}

@_cdecl("mps_svgf_set_normal_weight")
public func mps_svgf_set_normal_weight(_ handle: UnsafeMutableRawPointer?, _ value: Float) {
    guard let svgf: MPSSVGF = mps_borrow(handle) else { return }
    svgf.normalWeight = value
}

@_cdecl("mps_svgf_luminance_weight")
public func mps_svgf_luminance_weight(_ handle: UnsafeMutableRawPointer?) -> Float {
    guard let svgf: MPSSVGF = mps_borrow(handle) else { return 0 }
    return svgf.luminanceWeight
}

@_cdecl("mps_svgf_set_luminance_weight")
public func mps_svgf_set_luminance_weight(_ handle: UnsafeMutableRawPointer?, _ value: Float) {
    guard let svgf: MPSSVGF = mps_borrow(handle) else { return }
    svgf.luminanceWeight = value
}

@_cdecl("mps_svgf_channel_count")
public func mps_svgf_channel_count(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let svgf: MPSSVGF = mps_borrow(handle) else { return 0 }
    return svgf.channelCount
}

@_cdecl("mps_svgf_set_channel_count")
public func mps_svgf_set_channel_count(_ handle: UnsafeMutableRawPointer?, _ value: Int) {
    guard let svgf: MPSSVGF = mps_borrow(handle) else { return }
    svgf.channelCount = value
}

@_cdecl("mps_svgf_channel_count2")
public func mps_svgf_channel_count2(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let svgf: MPSSVGF = mps_borrow(handle) else { return 0 }
    return svgf.channelCount2
}

@_cdecl("mps_svgf_set_channel_count2")
public func mps_svgf_set_channel_count2(_ handle: UnsafeMutableRawPointer?, _ value: Int) {
    guard let svgf: MPSSVGF = mps_borrow(handle) else { return }
    svgf.channelCount2 = value
}
