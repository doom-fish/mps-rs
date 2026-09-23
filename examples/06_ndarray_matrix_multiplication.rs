use apple_metal::{resource_options, MetalBuffer, MetalDevice};
use apple_mps::{data_type, NDArray, NDArrayDescriptor, NDArrayMatrixMultiplication};

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

fn buffer_with_f32_values_padded(
    device: &MetalDevice,
    values: &[f32],
    byte_len: usize,
) -> MetalBuffer {
    let buffer = device
        .new_buffer(
            byte_len.max(core::mem::size_of_val(values)),
            resource_options::STORAGE_MODE_SHARED,
        )
        .expect("buffer");
    unsafe { buffer.write_bytes(0, as_bytes(values)) }.expect("write buffer");
    buffer
}

fn read_f32_values(buffer: &MetalBuffer, len: usize) -> Vec<f32> {
    let mut bytes = vec![0_u8; len * core::mem::size_of::<f32>()];
    unsafe { buffer.read_bytes(0, &mut bytes) }.expect("read buffer");
    bytes
        .chunks_exact(core::mem::size_of::<f32>())
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device.new_command_queue().expect("command queue");

    let descriptor = NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[2, 2, 1, 1])
        .expect("descriptor");
    let template = NDArray::new(&device, &descriptor).expect("template ndarray");
    let byte_len = template.resource_size();
    let rows = descriptor.length_of_dimension(1);
    let row_stride_floats = byte_len / core::mem::size_of::<f32>() / rows;
    let left_buffer =
        buffer_with_f32_values_padded(&device, &[1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0], byte_len);
    let right_buffer =
        buffer_with_f32_values_padded(&device, &[5.0, 6.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0], byte_len);
    let destination_buffer = buffer_with_f32_values_padded(&device, &[0.0; 8], byte_len);

    let left = NDArray::new_with_buffer(&left_buffer, 0, &descriptor).expect("left ndarray");
    let right = NDArray::new_with_buffer(&right_buffer, 0, &descriptor).expect("right ndarray");
    let destination =
        NDArray::new_with_buffer(&destination_buffer, 0, &descriptor).expect("destination ndarray");

    let kernel = NDArrayMatrixMultiplication::new(&device, 2).expect("ndarray matmul");
    kernel.set_alpha(1.0);
    kernel.set_beta(0.0);

    let command_buffer = queue.new_command_buffer().expect("command buffer");
    kernel
        .encode_to_destination(&command_buffer, &[&left, &right], &destination)
        .expect("encode ndarray matmul");
    command_buffer.commit().expect("commit");
    command_buffer
        .wait_until_completed()
        .expect("command buffer completed");

    let padded_output = read_f32_values(&destination_buffer, row_stride_floats * rows);
    let output = [
        padded_output[0],
        padded_output[1],
        padded_output[row_stride_floats],
        padded_output[row_stride_floats + 1],
    ];
    println!("{output:?}");
}
