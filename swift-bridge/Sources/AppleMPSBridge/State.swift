import Metal
import MetalPerformanceShaders

@_cdecl("mps_state_resource_list_new")
public func mps_state_resource_list_new() -> UnsafeMutableRawPointer? {
    mps_retain(MPSStateResourceList())
}

@_cdecl("mps_state_resource_list_append_buffer")
public func mps_state_resource_list_append_buffer(
    _ handle: UnsafeMutableRawPointer?,
    _ size: Int
) {
    guard let resourceList: MPSStateResourceList = mps_borrow(handle) else { return }
    resourceList.appendBuffer(size)
}

@_cdecl("mps_state_temporary_new")
public func mps_state_temporary_new(
    _ commandBufferHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle) else { return nil }
    return mps_retain(MPSState.temporaryState(with: commandBuffer))
}

@_cdecl("mps_state_temporary_new_with_buffer_size")
public func mps_state_temporary_new_with_buffer_size(
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ bufferSize: Int
) -> UnsafeMutableRawPointer? {
    guard let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle) else { return nil }
    return mps_retain(MPSState.temporaryState(with: commandBuffer, bufferSize: bufferSize))
}

@_cdecl("mps_state_new_with_buffer_size")
public func mps_state_new_with_buffer_size(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ bufferSize: Int
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSState(device: device, bufferSize: bufferSize))
}

@_cdecl("mps_state_new_with_resource_list")
public func mps_state_new_with_resource_list(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ resourceListHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle),
          let resourceList: MPSStateResourceList = mps_borrow(resourceListHandle)
    else {
        return nil
    }
    return mps_retain(MPSState(device: device, resourceList: resourceList))
}

@_cdecl("mps_state_temporary_new_with_resource_list")
public func mps_state_temporary_new_with_resource_list(
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ resourceListHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let resourceList: MPSStateResourceList = mps_borrow(resourceListHandle)
    else {
        return nil
    }
    return mps_retain(MPSState.temporaryState(with: commandBuffer, resourceList: resourceList))
}

@_cdecl("mps_state_resource_count")
public func mps_state_resource_count(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let state: MPSState = mps_borrow(handle) else { return 0 }
    return state.resourceCount
}

@_cdecl("mps_state_read_count")
public func mps_state_read_count(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let state: MPSState = mps_borrow(handle) else { return 0 }
    return state.readCount
}

@_cdecl("mps_state_set_read_count")
public func mps_state_set_read_count(_ handle: UnsafeMutableRawPointer?, _ value: Int) {
    guard let state: MPSState = mps_borrow(handle) else { return }
    state.readCount = value
}

@_cdecl("mps_state_is_temporary")
public func mps_state_is_temporary(_ handle: UnsafeMutableRawPointer?) -> Bool {
    guard let state: MPSState = mps_borrow(handle) else { return false }
    return state.isTemporary
}

@_cdecl("mps_state_buffer_size_at_index")
public func mps_state_buffer_size_at_index(_ handle: UnsafeMutableRawPointer?, _ index: Int) -> Int {
    guard let state: MPSState = mps_borrow(handle) else { return 0 }
    return state.bufferSize(at: index)
}

@_cdecl("mps_state_texture_info")
public func mps_state_texture_info(
    _ handle: UnsafeMutableRawPointer?,
    _ index: Int,
    _ width: UnsafeMutablePointer<Int>?,
    _ height: UnsafeMutablePointer<Int>?,
    _ depth: UnsafeMutablePointer<Int>?,
    _ arrayLength: UnsafeMutablePointer<Int>?,
    _ pixelFormat: UnsafeMutablePointer<UInt>?,
    _ textureType: UnsafeMutablePointer<UInt>?,
    _ usage: UnsafeMutablePointer<UInt>?
) {
    guard let state: MPSState = mps_borrow(handle) else { return }
    let info = state.textureInfo(at: index)
    width?.pointee = info.width
    height?.pointee = info.height
    depth?.pointee = info.depth
    arrayLength?.pointee = info.arrayLength
    pixelFormat?.pointee = info.pixelFormat.rawValue
    textureType?.pointee = info.textureType.rawValue
    usage?.pointee = info.usage.rawValue
}

@_cdecl("mps_state_resource_type_at_index")
public func mps_state_resource_type_at_index(_ handle: UnsafeMutableRawPointer?, _ index: Int) -> UInt {
    guard let state: MPSState = mps_borrow(handle) else { return 0 }
    return state.resourceType(at: index).rawValue
}

@_cdecl("mps_state_synchronize_on_command_buffer")
public func mps_state_synchronize_on_command_buffer(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?
) {
    guard let state: MPSState = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle)
    else {
        return
    }
    state.synchronize(on: commandBuffer)
}

@_cdecl("mps_state_resource_size")
public func mps_state_resource_size(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let state: MPSState = mps_borrow(handle) else { return 0 }
    return state.resourceSize()
}

@_cdecl("mps_state_batch_increment_read_count")
public func mps_state_batch_increment_read_count(
    _ handles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ count: Int,
    _ amount: Int
) -> Int {
    guard let states: [MPSState] = mps_borrow_array(handles, count: count) else { return 0 }
    return MPSStateBatchIncrementReadCount(states, amount)
}

@_cdecl("mps_state_batch_synchronize")
public func mps_state_batch_synchronize(
    _ handles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ count: Int,
    _ commandBufferHandle: UnsafeMutableRawPointer?
) {
    guard let states: [MPSState] = mps_borrow_array(handles, count: count),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle)
    else {
        return
    }
    if states.isEmpty { return }
    MPSStateBatchSynchronize(states, commandBuffer)
}

@_cdecl("mps_state_batch_resource_size")
public func mps_state_batch_resource_size(
    _ handles: UnsafePointer<UnsafeMutableRawPointer?>?,
    _ count: Int
) -> Int {
    guard let states: [MPSState] = mps_borrow_array(handles, count: count) else { return 0 }
    return MPSStateBatchResourceSize(states)
}
