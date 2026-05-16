use apple_metal::MetalDevice;
use apple_mps::{data_type, NDArray, NDArrayDescriptor, NDArrayIdentity};

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device.new_command_queue().expect("command queue");

    let descriptor =
        NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[4]).expect("descriptor");
    assert_eq!(descriptor.number_of_dimensions(), 1);
    assert_eq!(descriptor.length_of_dimension(0), 4);

    descriptor.reshape_with_dimension_sizes(&[2, 2]);
    assert_eq!(descriptor.number_of_dimensions(), 2);
    assert_eq!(descriptor.length_of_dimension(0), 2);
    assert_eq!(descriptor.length_of_dimension(1), 2);

    let array_descriptor = NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[2, 2])
        .expect("array descriptor");
    let array = NDArray::new(&device, &array_descriptor).expect("ndarray");
    assert_eq!(array.number_of_dimensions(), 2);
    assert_eq!(array.length_of_dimension(0), 2);
    assert!(array.resource_size() > 0);

    if let Some(identity) = NDArrayIdentity::new(&device) {
        let destination_descriptor =
            NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[4])
                .expect("destination descriptor");
        let destination =
            NDArray::new(&device, &destination_descriptor).expect("destination ndarray");
        let command_buffer = queue.new_command_buffer().expect("command buffer");
        assert!(identity.reshape_into(Some(&command_buffer), &array, &[4], &destination));
        command_buffer.commit();
        command_buffer.wait_until_completed();
        assert_eq!(destination.number_of_dimensions(), 1);
        assert_eq!(destination.length_of_dimension(0), 4);
        println!(
            "ndarray smoke passed: resource_size={} dims={}x{}",
            array.resource_size(),
            array.length_of_dimension(0),
            array.length_of_dimension(1)
        );
    } else {
        println!("ndarray smoke skipped: MPSNDArrayIdentity unavailable on this macOS version");
    }
}
