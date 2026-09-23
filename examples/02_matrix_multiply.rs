use apple_metal::{resource_options, MetalBuffer, MetalDevice};
use apple_mps::{data_type, Matrix, MatrixDescriptor, MatrixMultiplication};

fn as_bytes(values: &[f32]) -> &[u8] {
    // SAFETY: `f32` has no invalid bit patterns and `values` remains alive for the returned slice.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

fn read_f32s(buffer: &MetalBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0_u8; len * core::mem::size_of::<f32>()];
    // SAFETY: The command buffer that wrote this buffer has completed, so no GPU access overlaps the read.
    unsafe { buffer.read_bytes(0, &mut bytes) }.expect("buffer has CPU-visible storage");
    bytes
        .chunks_exact(core::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device
        .new_command_queue()
        .expect("failed to create command queue");

    let left_desc = MatrixDescriptor::contiguous(2, 3, data_type::FLOAT32)
        .expect("FLOAT32 matrix descriptor should be supported");
    let right_desc = MatrixDescriptor::contiguous(3, 2, data_type::FLOAT32)
        .expect("FLOAT32 matrix descriptor should be supported");
    let result_desc = MatrixDescriptor::contiguous(2, 2, data_type::FLOAT32)
        .expect("FLOAT32 matrix descriptor should be supported");

    let left_buffer = device
        .new_buffer(
            left_desc.matrix_bytes,
            resource_options::STORAGE_MODE_SHARED,
        )
        .expect("failed to allocate left buffer");
    let right_buffer = device
        .new_buffer(
            right_desc.matrix_bytes,
            resource_options::STORAGE_MODE_SHARED,
        )
        .expect("failed to allocate right buffer");
    let result_buffer = device
        .new_buffer(
            result_desc.matrix_bytes,
            resource_options::STORAGE_MODE_SHARED,
        )
        .expect("failed to allocate result buffer");

    let left_values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let right_values = [7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let zeros = [0.0_f32; 4];

    unsafe {
        left_buffer
            .write_bytes(0, as_bytes(&left_values))
            .expect("write left buffer");
        right_buffer
            .write_bytes(0, as_bytes(&right_values))
            .expect("write right buffer");
        result_buffer
            .write_bytes(0, as_bytes(&zeros))
            .expect("write result buffer");
    }

    let left =
        Matrix::new_with_buffer(&left_buffer, left_desc).expect("failed to wrap left matrix");
    let right =
        Matrix::new_with_buffer(&right_buffer, right_desc).expect("failed to wrap right matrix");
    let result =
        Matrix::new_with_buffer(&result_buffer, result_desc).expect("failed to wrap result matrix");

    let gemm = MatrixMultiplication::new_simple(&device, 2, 2, 3)
        .expect("failed to create matrix multiplication kernel");
    let command_buffer = queue
        .new_command_buffer()
        .expect("failed to allocate command buffer");
    gemm.encode(&command_buffer, &left, &right, &result)
        .expect("encode matrix multiplication");
    command_buffer.commit().expect("commit");
    command_buffer
        .wait_until_completed()
        .expect("command buffer completed");

    let output = read_f32s(&result_buffer, 4);
    let expected = [58.0_f32, 64.0, 139.0, 154.0];
    for (actual, expected_value) in output.iter().zip(expected) {
        assert!(
            (actual - expected_value).abs() < 1.0e-4,
            "unexpected matrix multiply result: {output:?}"
        );
    }

    println!("matmul smoke passed: {output:?}");
}
