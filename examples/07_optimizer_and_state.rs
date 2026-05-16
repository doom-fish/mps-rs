use apple_metal::{resource_options, MetalBuffer, MetalDevice};
use apple_mps::{
    data_type, nn_regularization_type, state_batch_increment_read_count, state_resource_type,
    NNOptimizerStochasticGradientDescent, State, StateResourceList, Vector, VectorDescriptor,
};

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

fn buffer_with_f32_values(device: &MetalDevice, values: &[f32]) -> MetalBuffer {
    let buffer = device
        .new_buffer(
            core::mem::size_of_val(values),
            resource_options::STORAGE_MODE_SHARED,
        )
        .expect("buffer");
    let _ = buffer.write_bytes(as_bytes(values));
    buffer
}

fn read_f32_values(buffer: &MetalBuffer, len: usize) -> Vec<f32> {
    let ptr = buffer.contents().expect("buffer contents").cast::<f32>();
    unsafe { core::slice::from_raw_parts(ptr, len).to_vec() }
}

fn vector_with_values(device: &MetalDevice, values: &[f32]) -> (MetalBuffer, Vector) {
    let buffer = buffer_with_f32_values(device, values);
    let descriptor = VectorDescriptor::contiguous(values.len(), data_type::FLOAT32).expect("vector desc");
    let vector = Vector::new_with_buffer(&buffer, descriptor).expect("vector");
    (buffer, vector)
}

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device.new_command_queue().expect("command queue");

    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let temporary_a = State::temporary_with_buffer_size(&command_buffer, 32).expect("temporary state a");
    let temporary_b = State::temporary_with_buffer_size(&command_buffer, 64).expect("temporary state b");
    let unique_count = state_batch_increment_read_count(&[&temporary_a, &temporary_a, &temporary_b], 1);
    assert_eq!(unique_count, 2);
    assert_eq!(temporary_a.resource_type_at_index(0), state_resource_type::BUFFER);
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let resource_list = StateResourceList::new().expect("resource list");
    resource_list.append_buffer(16);
    let persistent_state = State::new_with_resource_list(&device, &resource_list).expect("persistent state");
    assert_eq!(persistent_state.resource_count(), 1);
    assert_eq!(persistent_state.buffer_size_at_index(0), 16);

    let (gradient_buffer, gradient_vector) = vector_with_values(&device, &[0.1, -0.2]);
    let (values_buffer, values_vector) = vector_with_values(&device, &[1.0, -1.0]);
    let (result_buffer, result_vector) = vector_with_values(&device, &[0.0, 0.0]);
    let optimizer = NNOptimizerStochasticGradientDescent::new(&device, 0.5).expect("sgd");
    let base = optimizer.as_optimizer().expect("optimizer base");
    assert_eq!(base.regularization_type(), nn_regularization_type::NONE);

    let command_buffer = queue.new_command_buffer().expect("command buffer");
    optimizer.encode_vector(
        &command_buffer,
        &gradient_vector,
        &values_vector,
        None,
        &result_vector,
    );
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let _ = gradient_buffer;
    let _ = values_buffer;
    let output = read_f32_values(&result_buffer, 2);
    println!("{output:?}");
}
