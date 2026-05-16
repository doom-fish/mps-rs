import Metal
import MetalPerformanceShaders

@_cdecl("mps_predicate_new_with_buffer")
public func mps_predicate_new_with_buffer(
    _ bufferHandle: UnsafeMutableRawPointer?,
    _ offset: Int
) -> UnsafeMutableRawPointer? {
    guard let buffer: MTLBuffer = mps_borrow(bufferHandle) else { return nil }
    return mps_retain(MPSPredicate(buffer: buffer, offset: offset))
}

@_cdecl("mps_predicate_new_with_device")
public func mps_predicate_new_with_device(
    _ deviceHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(MPSPredicate(device: device))
}

@_cdecl("mps_predicate_offset")
public func mps_predicate_offset(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let predicate: MPSPredicate = mps_borrow(handle) else { return 0 }
    return predicate.predicateOffset
}

@_cdecl("mps_command_buffer_new_with_command_buffer")
public func mps_command_buffer_new_with_command_buffer(
    _ commandBufferHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle) else { return nil }
    return mps_retain(MPSCommandBuffer(commandBuffer: commandBuffer))
}

@_cdecl("mps_command_buffer_from_command_queue")
public func mps_command_buffer_from_command_queue(
    _ commandQueueHandle: UnsafeMutableRawPointer?
) -> UnsafeMutableRawPointer? {
    guard let commandQueue: MTLCommandQueue = mps_borrow(commandQueueHandle),
          let commandBuffer = commandQueue.makeCommandBuffer()
    else {
        return nil
    }
    return mps_retain(MPSCommandBuffer(commandBuffer: commandBuffer))
}

@_cdecl("mps_command_buffer_set_predicate")
public func mps_command_buffer_set_predicate(
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ predicateHandle: UnsafeMutableRawPointer?
) {
    guard let commandBuffer: MPSCommandBuffer = mps_borrow(commandBufferHandle),
          let predicate: MPSPredicate = mps_borrow(predicateHandle)
    else {
        return
    }
    commandBuffer.predicate = predicate
}

@_cdecl("mps_command_buffer_clear_predicate")
public func mps_command_buffer_clear_predicate(_ commandBufferHandle: UnsafeMutableRawPointer?) {
    guard let commandBuffer: MPSCommandBuffer = mps_borrow(commandBufferHandle) else { return }
    commandBuffer.predicate = nil
}

@_cdecl("mps_command_buffer_prefetch_heap")
public func mps_command_buffer_prefetch_heap(
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ size: Int
) {
    guard let commandBuffer: MPSCommandBuffer = mps_borrow(commandBufferHandle) else { return }
    commandBuffer.prefetchHeap(forWorkloadSize: size)
}

@_cdecl("mps_command_buffer_commit_and_continue")
public func mps_command_buffer_commit_and_continue(
    _ commandBufferHandle: UnsafeMutableRawPointer?
) {
    guard let commandBuffer: MPSCommandBuffer = mps_borrow(commandBufferHandle) else { return }
    commandBuffer.commitAndContinue()
}
