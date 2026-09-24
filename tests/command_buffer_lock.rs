use std::sync::Barrier;
use std::thread;
use std::time::{Duration, Instant};

use apple_metal::{
    pixel_format, resource_options, texture_usage, CommandBufferError, MetalDevice, MetalTexture,
    TextureDescriptor,
};
use apple_mps::{
    data_type, hint_temporary_memory_high_water_mark, Error, ImageGaussianBlur, Matrix,
    MatrixDescriptor, MatrixMultiplication, MpsCommandBuffer, State,
};

fn device() -> MetalDevice {
    MetalDevice::system_default().expect("Metal device")
}

fn texture(device: &MetalDevice) -> MetalTexture {
    device
        .new_texture(TextureDescriptor {
            usage: texture_usage::SHADER_READ | texture_usage::SHADER_WRITE,
            ..TextureDescriptor::new_2d(8, 8, pixel_format::RGBA8UNORM)
        })
        .expect("texture")
}

const fn active_encoder() -> Error {
    Error::CommandBuffer(CommandBufferError::ActiveEncoder)
}

#[test]
fn open_apple_metal_encoders_are_refused_instead_of_aborting() {
    let device = device();
    let queue = device.new_command_queue().expect("command queue");
    let blur = ImageGaussianBlur::new(&device, 1.0).expect("blur");
    let source = texture(&device);
    let destination = texture(&device);
    let command_buffer = queue.new_command_buffer().expect("command buffer");

    let encoder = command_buffer
        .new_compute_command_encoder()
        .expect("compute encoder");
    assert_eq!(
        blur.encode_texture(&command_buffer, &source, &destination),
        Err(active_encoder())
    );
    assert_eq!(
        hint_temporary_memory_high_water_mark(&command_buffer, 1 << 20),
        Err(active_encoder())
    );
    assert!(State::temporary(&command_buffer).is_none());
    let wrapped =
        MpsCommandBuffer::new_with_command_buffer(&command_buffer).expect("MPS command buffer");
    assert_eq!(
        wrapped.prefetch_heap_for_workload_size(1 << 16),
        Err(active_encoder())
    );
    encoder.end_encoding().expect("end encoding");

    blur.encode_texture(&command_buffer, &source, &destination)
        .expect("encode after the encoder ended");
    wrapped
        .prefetch_heap_for_workload_size(1 << 16)
        .expect("prefetch after the encoder ended");
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
}

#[test]
fn commits_from_other_threads_never_interleave_with_mps_encodes() {
    let device = device();
    let queue = device.new_command_queue().expect("command queue");
    let blur = ImageGaussianBlur::new(&device, 1.0).expect("blur");
    let source = texture(&device);
    let destination = texture(&device);
    let (mut encoded, mut refused) = (0, 0);

    for delay in (0..4096).map(|iteration| iteration % 256) {
        let command_buffer = queue.new_command_buffer().expect("command buffer");
        let committer = command_buffer.clone();
        let barrier = Barrier::new(2);
        let (encode, commit) = thread::scope(|scope| {
            let commit = scope.spawn(|| {
                barrier.wait();
                let start = Instant::now();
                while start.elapsed() < Duration::from_nanos(delay * 250) {
                    std::hint::spin_loop();
                }
                committer.commit()
            });
            barrier.wait();
            let encode = blur.encode_texture(&command_buffer, &source, &destination);
            (encode, commit.join().expect("commit thread"))
        });
        match (encode, commit) {
            (Ok(()), Ok(())) => encoded += 1,
            (Ok(()), Err(CommandBufferError::ActiveEncoder)) => {
                encoded += 1;
                command_buffer.commit().expect("commit after the encode");
            }
            (Err(Error::CommandBuffer(CommandBufferError::InvalidState { .. })), Ok(())) => {
                refused += 1;
            }
            other => panic!("unexpected outcome {other:?}"),
        }
        command_buffer.wait_until_completed().expect("completed");
    }
    assert_eq!(encoded + refused, 4096);
}

#[test]
fn queue_created_mps_command_buffers_can_be_committed() {
    let device = device();
    let queue = device.new_command_queue().expect("command queue");
    let wrapped = MpsCommandBuffer::from_command_queue(&queue).expect("MPS command buffer");
    let descriptor = MatrixDescriptor::contiguous(2, 2, data_type::FLOAT32).expect("descriptor");
    let matrix = |values: [f32; 4]| {
        let buffer = device
            .new_buffer_with_bytes(
                &values.map(f32::to_ne_bytes).concat(),
                resource_options::STORAGE_MODE_SHARED,
            )
            .expect("buffer");
        let matrix = Matrix::new_with_buffer(&buffer, descriptor).expect("matrix");
        (buffer, matrix)
    };
    let (_left_buffer, left) = matrix([1.0, 2.0, 3.0, 4.0]);
    let (_right_buffer, right) = matrix([1.0, 0.0, 0.0, 1.0]);
    let (result_buffer, result) = matrix([0.0; 4]);
    let multiplication = MatrixMultiplication::new_simple(&device, 2, 2, 2).expect("kernel");

    multiplication
        .encode(wrapped.command_buffer(), &left, &right, &result)
        .expect("encode");
    wrapped.command_buffer().commit().expect("commit");
    wrapped
        .command_buffer()
        .wait_until_completed()
        .expect("completed");
    let mut bytes = [0_u8; 16];
    unsafe { result_buffer.read_bytes(0, &mut bytes) }.expect("shared storage");
    let values: Vec<f32> = bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes(chunk.try_into().expect("word")))
        .collect();
    assert_eq!(values, [1.0, 2.0, 3.0, 4.0]);
}
