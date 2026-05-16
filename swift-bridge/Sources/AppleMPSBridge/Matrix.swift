import Metal
import MetalPerformanceShaders

@_cdecl("mps_matrix_descriptor_row_bytes_for_columns")
public func mps_matrix_descriptor_row_bytes_for_columns(
    _ columns: Int,
    _ dataTypeRaw: UInt32
) -> Int {
    guard let dataType = mps_data_type(dataTypeRaw) else { return 0 }
    return MPSMatrixDescriptor.rowBytes(forColumns: columns, dataType: dataType)
}

@_cdecl("mps_vector_descriptor_vector_bytes_for_length")
public func mps_vector_descriptor_vector_bytes_for_length(
    _ length: Int,
    _ dataTypeRaw: UInt32
) -> Int {
    guard let dataType = mps_data_type(dataTypeRaw) else { return 0 }
    return MPSVectorDescriptor.vectorBytes(forLength: length, dataType: dataType)
}

@_cdecl("mps_matrix_new_with_buffer")
public func mps_matrix_new_with_buffer(
    _ bufferHandle: UnsafeMutableRawPointer?,
    _ rows: Int,
    _ columns: Int,
    _ matrices: Int,
    _ rowBytes: Int,
    _ matrixBytes: Int,
    _ dataTypeRaw: UInt32
) -> UnsafeMutableRawPointer? {
    guard let buffer: MTLBuffer = mps_borrow(bufferHandle),
          let dataType = mps_data_type(dataTypeRaw)
    else {
        return nil
    }

    let descriptor: MPSMatrixDescriptor
    if matrices == 1 {
        descriptor = MPSMatrixDescriptor(
            rows: rows,
            columns: columns,
            rowBytes: rowBytes,
            dataType: dataType
        )
    } else {
        descriptor = MPSMatrixDescriptor(
            rows: rows,
            columns: columns,
            matrices: matrices,
            rowBytes: rowBytes,
            matrixBytes: matrixBytes,
            dataType: dataType
        )
    }
    return mps_retain(MPSMatrix(buffer: buffer, descriptor: descriptor))
}

@_cdecl("mps_matrix_rows")
public func mps_matrix_rows(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let matrix: MPSMatrix = mps_borrow(handle) else { return 0 }
    return matrix.rows
}

@_cdecl("mps_matrix_columns")
public func mps_matrix_columns(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let matrix: MPSMatrix = mps_borrow(handle) else { return 0 }
    return matrix.columns
}

@_cdecl("mps_matrix_matrices")
public func mps_matrix_matrices(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let matrix: MPSMatrix = mps_borrow(handle) else { return 0 }
    return matrix.matrices
}

@_cdecl("mps_matrix_row_bytes")
public func mps_matrix_row_bytes(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let matrix: MPSMatrix = mps_borrow(handle) else { return 0 }
    return matrix.rowBytes
}

@_cdecl("mps_matrix_matrix_bytes")
public func mps_matrix_matrix_bytes(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let matrix: MPSMatrix = mps_borrow(handle) else { return 0 }
    return matrix.matrixBytes
}

@_cdecl("mps_matrix_data_type")
public func mps_matrix_data_type(_ handle: UnsafeMutableRawPointer?) -> UInt32 {
    guard let matrix: MPSMatrix = mps_borrow(handle) else { return 0 }
    return matrix.dataType.rawValue
}

@_cdecl("mps_vector_new_with_buffer")
public func mps_vector_new_with_buffer(
    _ bufferHandle: UnsafeMutableRawPointer?,
    _ length: Int,
    _ vectors: Int,
    _ vectorBytes: Int,
    _ dataTypeRaw: UInt32
) -> UnsafeMutableRawPointer? {
    guard let buffer: MTLBuffer = mps_borrow(bufferHandle),
          let dataType = mps_data_type(dataTypeRaw)
    else {
        return nil
    }

    let descriptor: MPSVectorDescriptor
    if vectors == 1 {
        descriptor = MPSVectorDescriptor(length: length, dataType: dataType)
    } else {
        descriptor = MPSVectorDescriptor(
            length: length,
            vectors: vectors,
            vectorBytes: vectorBytes,
            dataType: dataType
        )
    }
    return mps_retain(MPSVector(buffer: buffer, descriptor: descriptor))
}

@_cdecl("mps_vector_length")
public func mps_vector_length(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let vector: MPSVector = mps_borrow(handle) else { return 0 }
    return vector.length
}

@_cdecl("mps_vector_vectors")
public func mps_vector_vectors(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let vector: MPSVector = mps_borrow(handle) else { return 0 }
    return vector.vectors
}

@_cdecl("mps_vector_vector_bytes")
public func mps_vector_vector_bytes(_ handle: UnsafeMutableRawPointer?) -> Int {
    guard let vector: MPSVector = mps_borrow(handle) else { return 0 }
    return vector.vectorBytes
}

@_cdecl("mps_vector_data_type")
public func mps_vector_data_type(_ handle: UnsafeMutableRawPointer?) -> UInt32 {
    guard let vector: MPSVector = mps_borrow(handle) else { return 0 }
    return vector.dataType.rawValue
}

@_cdecl("mps_matrix_multiplication_new")
public func mps_matrix_multiplication_new(
    _ deviceHandle: UnsafeMutableRawPointer?,
    _ transposeLeft: Bool,
    _ transposeRight: Bool,
    _ resultRows: Int,
    _ resultColumns: Int,
    _ interiorColumns: Int,
    _ alpha: Double,
    _ beta: Double
) -> UnsafeMutableRawPointer? {
    guard let device: MTLDevice = mps_borrow(deviceHandle) else { return nil }
    return mps_retain(
        MPSMatrixMultiplication(
            device: device,
            transposeLeft: transposeLeft,
            transposeRight: transposeRight,
            resultRows: resultRows,
            resultColumns: resultColumns,
            interiorColumns: interiorColumns,
            alpha: alpha,
            beta: beta
        )
    )
}

@_cdecl("mps_matrix_multiplication_encode")
public func mps_matrix_multiplication_encode(
    _ handle: UnsafeMutableRawPointer?,
    _ commandBufferHandle: UnsafeMutableRawPointer?,
    _ leftMatrixHandle: UnsafeMutableRawPointer?,
    _ rightMatrixHandle: UnsafeMutableRawPointer?,
    _ resultMatrixHandle: UnsafeMutableRawPointer?
) {
    guard let kernel: MPSMatrixMultiplication = mps_borrow(handle),
          let commandBuffer: MTLCommandBuffer = mps_borrow(commandBufferHandle),
          let leftMatrix: MPSMatrix = mps_borrow(leftMatrixHandle),
          let rightMatrix: MPSMatrix = mps_borrow(rightMatrixHandle),
          let resultMatrix: MPSMatrix = mps_borrow(resultMatrixHandle)
    else {
        return
    }

    kernel.encode(
        commandBuffer: commandBuffer,
        leftMatrix: leftMatrix,
        rightMatrix: rightMatrix,
        resultMatrix: resultMatrix
    )
}
