use apple_metal::{resource_options, MetalBuffer, MetalDevice};
use apple_mps::{
    data_type, Error, Matrix, MatrixDescriptor, MatrixMultiplication,
    MatrixMultiplicationDescriptor, NDArray, NDArrayDescriptor, NDArrayIdentity,
    NDArrayMatrixMultiplication, Vector, VectorDescriptor,
};

fn device() -> MetalDevice {
    MetalDevice::system_default().expect("no Metal device available")
}

fn buffer(device: &MetalDevice, length: usize) -> MetalBuffer {
    device
        .new_buffer(length, resource_options::STORAGE_MODE_SHARED)
        .expect("buffer")
}

fn filled(device: &MetalDevice, values: &[f32], length: usize) -> MetalBuffer {
    let buffer = buffer(device, length);
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    unsafe { buffer.write_bytes(0, &bytes) }.expect("write buffer");
    buffer
}

fn read(buffer: &MetalBuffer, count: usize) -> Vec<f32> {
    let mut bytes = vec![0_u8; count * 4];
    unsafe { buffer.read_bytes(0, &mut bytes) }.expect("read buffer");
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn matrix(device: &MetalDevice, rows: usize, columns: usize, data: u32) -> (MetalBuffer, Matrix) {
    let descriptor = MatrixDescriptor::contiguous(rows, columns, data).expect("descriptor");
    let buffer = buffer(device, descriptor.matrix_bytes.max(4));
    let matrix = Matrix::new_with_buffer(&buffer, descriptor).expect("matrix");
    (buffer, matrix)
}

fn ndarray(device: &MetalDevice, sizes: &[usize]) -> NDArray {
    let descriptor =
        NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, sizes).expect("descriptor");
    NDArray::new(device, &descriptor).expect("ndarray")
}

#[test]
fn matrices_larger_than_their_buffer_are_refused() {
    let device = device();
    let small = buffer(&device, 4);
    let descriptor = MatrixDescriptor::contiguous(1024, 1024, data_type::FLOAT32).expect("desc");
    assert_eq!(descriptor.required_buffer_length(), Ok(4_194_304));
    assert_eq!(
        Matrix::new_with_buffer(&small, descriptor).err(),
        Some(Error::BufferTooSmall {
            required: 4_194_304,
            length: 4,
        })
    );

    let batched = MatrixDescriptor::with_strides(2, 3, 2, 16, 48, data_type::FLOAT32);
    assert_eq!(batched.required_buffer_length(), Ok(76));
    assert!(Matrix::new_with_buffer(&buffer(&device, 75), batched).is_err());
    let matrix = Matrix::new_with_buffer(&buffer(&device, 76), batched).expect("exact fit");
    assert_eq!(matrix.matrices(), 2);
    assert_eq!(matrix.row_bytes(), 16);
    assert_eq!(matrix.matrix_bytes(), 48);
}

#[test]
fn matrix_strides_follow_the_mps_rules() {
    let device = device();
    let large = buffer(&device, 4096);
    let check = |descriptor: MatrixDescriptor| Matrix::new_with_buffer(&large, descriptor).err();
    assert_eq!(
        check(MatrixDescriptor::with_strides(
            2,
            2,
            1,
            9,
            18,
            data_type::FLOAT32
        )),
        Some(Error::Misaligned {
            field: "row_bytes",
            value: 9,
            alignment: 4,
        })
    );
    assert_eq!(
        check(MatrixDescriptor::with_strides(
            2,
            2,
            1,
            4,
            8,
            data_type::FLOAT32
        )),
        Some(Error::DimensionMismatch {
            field: "row_bytes",
            expected: 8,
            actual: 4,
        })
    );
    assert_eq!(
        check(MatrixDescriptor::with_strides(
            2,
            2,
            2,
            8,
            12,
            data_type::FLOAT32
        )),
        Some(Error::DimensionMismatch {
            field: "matrix_bytes",
            expected: 16,
            actual: 12,
        })
    );
    assert_eq!(
        check(MatrixDescriptor::with_strides(
            2,
            2,
            2,
            8,
            20,
            data_type::FLOAT32
        )),
        Some(Error::Misaligned {
            field: "matrix_bytes",
            value: 20,
            alignment: 8,
        })
    );
    assert_eq!(
        check(MatrixDescriptor::with_strides(2, 2, 1, 8, 16, 0x1234)),
        Some(Error::UnsupportedDataType(0x1234))
    );
    assert_eq!(
        check(MatrixDescriptor::with_strides(
            usize::MAX,
            0,
            1,
            0,
            0,
            data_type::FLOAT32
        )),
        Some(Error::Overflow)
    );
    assert_eq!(
        check(MatrixDescriptor::with_strides(
            usize::MAX / 2,
            1,
            1,
            8,
            0,
            data_type::FLOAT32
        )),
        Some(Error::Overflow)
    );
    let bf16 = MatrixDescriptor::contiguous(2, 2, data_type::BFLOAT16).expect("bf16 desc");
    assert_eq!(bf16.row_bytes, 4);
    assert!(Matrix::new_with_buffer(&large, bf16).is_ok());
}

#[test]
fn vectors_are_checked_against_their_buffer() {
    let device = device();
    let batched = VectorDescriptor::with_stride(3, 2, 16, data_type::FLOAT32);
    assert_eq!(batched.required_buffer_length(), Ok(28));
    assert_eq!(
        Vector::new_with_buffer(&buffer(&device, 27), batched).err(),
        Some(Error::BufferTooSmall {
            required: 28,
            length: 27,
        })
    );
    assert_eq!(
        Vector::new_with_buffer(&buffer(&device, 28), batched)
            .expect("exact fit")
            .vectors(),
        2
    );
    let large = buffer(&device, 4096);
    assert_eq!(
        Vector::new_with_buffer(
            &large,
            VectorDescriptor::with_stride(2, 2, 9, data_type::FLOAT32)
        )
        .err(),
        Some(Error::Misaligned {
            field: "vector_bytes",
            value: 9,
            alignment: 4,
        })
    );
    assert_eq!(
        Vector::new_with_buffer(
            &large,
            VectorDescriptor::with_stride(4, 2, 8, data_type::FLOAT32)
        )
        .err(),
        Some(Error::DimensionMismatch {
            field: "vector_bytes",
            expected: 16,
            actual: 8,
        })
    );
    let single = VectorDescriptor::contiguous(1024, data_type::FLOAT32).expect("desc");
    assert_eq!(
        Vector::new_with_buffer(&buffer(&device, 4), single).err(),
        Some(Error::BufferTooSmall {
            required: 4096,
            length: 4,
        })
    );
}

#[test]
fn buffer_backed_ndarrays_are_checked_against_their_buffer() {
    let device = device();
    let huge = NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[1024, 1024])
        .expect("descriptor");
    assert_eq!(huge.required_buffer_length(), Ok(4_194_304));
    assert_eq!(
        NDArray::new_with_buffer(&buffer(&device, 4), 0, &huge).err(),
        Some(Error::BufferTooSmall {
            required: 4_194_304,
            length: 4,
        })
    );
    let padded =
        NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[3, 5]).expect("descriptor");
    assert_eq!(padded.required_buffer_length(), Ok(80));
    assert_eq!(
        padded.required_buffer_length(),
        Ok(NDArray::new(&device, &padded)
            .expect("template")
            .resource_size())
    );
    let storage = buffer(&device, 96);
    assert_eq!(
        NDArray::new_with_buffer(&storage, 2, &padded).err(),
        Some(Error::Misaligned {
            field: "offset",
            value: 2,
            alignment: 4,
        })
    );
    assert_eq!(
        NDArray::new_with_buffer(&storage, 20, &padded).err(),
        Some(Error::BufferTooSmall {
            required: 100,
            length: 96,
        })
    );
    let array = NDArray::new_with_buffer(&storage, 16, &padded).expect("fits at offset 16");
    assert_eq!(array.length_of_dimension(0), 3);
    assert_eq!(array.length_of_dimension(1), 5);
    assert_eq!(array.length_of_dimension(7), 1);
    let packed =
        NDArrayDescriptor::with_dimension_sizes(data_type::INT4, &[8]).expect("int4 descriptor");
    assert_eq!(
        NDArray::new_with_buffer(&storage, 0, &packed).err(),
        Some(Error::UnsupportedDataType(data_type::INT4))
    );
}

#[test]
fn ndarray_descriptors_reject_shapes_mps_aborts_on() {
    assert!(NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[]).is_none());
    assert!(NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[1; 17]).is_none());
    assert!(
        NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[1 << 16, 1 << 15]).is_none()
    );
    assert!(NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[usize::MAX]).is_none());
    let descriptor =
        NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[2, 3]).expect("descriptor");
    assert!(matches!(
        descriptor.set_number_of_dimensions(17),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        descriptor.transpose_dimension(0, 5),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        descriptor.reshape_with_dimension_sizes(&[1; 17]),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(descriptor.number_of_dimensions(), 2);
    descriptor.transpose_dimension(0, 1).expect("transpose");
    assert_eq!(descriptor.length_of_dimension(0), 3);
    assert_eq!(descriptor.length_of_dimension(1), 2);
    descriptor
        .reshape_with_dimension_sizes(&[6])
        .expect("reshape");
    assert_eq!(descriptor.number_of_dimensions(), 1);
    assert_eq!(descriptor.length_of_dimension(0), 6);
    descriptor.set_number_of_dimensions(3).expect("rank 3");
    assert_eq!(descriptor.length_of_dimension(2), 1);
}

#[test]
fn identity_reshapes_must_preserve_the_volume() {
    let device = device();
    let Some(identity) = NDArrayIdentity::new(&device) else {
        return;
    };
    let queue = device.new_command_queue().expect("queue");
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let source = ndarray(&device, &[2, 2]);
    assert!(identity
        .reshape_with_command_buffer(&command_buffer, &source, &[5])
        .is_none());
    assert!(identity
        .reshape_with_command_buffer(&command_buffer, &source, &[])
        .is_none());
    assert!(!identity.reshape_into(
        Some(&command_buffer),
        &source,
        &[4],
        &ndarray(&device, &[7])
    ));
    assert!(!identity.reshape_into(
        Some(&command_buffer),
        &source,
        &[4],
        &ndarray(&device, &[4, 1])
    ));
    let six = ndarray(&device, &[6]);
    assert!(!identity.reshape_into(
        Some(&command_buffer),
        &six,
        &[3, 2],
        &ndarray(&device, &[3, 2])
    ));
    let destination = ndarray(&device, &[2, 3]);
    assert!(identity.reshape_into(Some(&command_buffer), &six, &[3, 2], &destination));
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
    assert_eq!(destination.length_of_dimension(0), 2);
    assert_eq!(destination.length_of_dimension(1), 3);
}

#[test]
fn ndarray_multiplication_checks_sources_and_shapes() {
    let device = device();
    assert!(NDArrayMatrixMultiplication::new(&device, 0).is_none());
    assert!(NDArrayMatrixMultiplication::new(&device, 1).is_none());
    assert!(NDArrayMatrixMultiplication::new(&device, 4).is_none());
    let kernel = NDArrayMatrixMultiplication::new(&device, 2).expect("kernel");
    assert_eq!(kernel.source_count(), 2);
    let queue = device.new_command_queue().expect("queue");
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let a = ndarray(&device, &[3, 2]);
    let b = ndarray(&device, &[4, 3]);
    assert!(kernel.encode(&command_buffer, &[&a]).is_none());
    assert!(kernel.encode(&command_buffer, &[&a, &b, &b]).is_none());
    assert_eq!(
        kernel.encode_to_destination(
            &command_buffer,
            &[&a, &ndarray(&device, &[4, 5])],
            &ndarray(&device, &[4, 2])
        ),
        Err(Error::DimensionMismatch {
            field: "interior dimension",
            expected: 3,
            actual: 5,
        })
    );
    assert_eq!(
        kernel.encode_to_destination(
            &command_buffer,
            &[&ndarray(&device, &[3, 2, 3]), &ndarray(&device, &[4, 3, 2])],
            &ndarray(&device, &[4, 2, 3])
        ),
        Err(Error::DimensionMismatch {
            field: "batch dimension",
            expected: 3,
            actual: 2,
        })
    );
    assert_eq!(
        kernel.encode_to_destination(&command_buffer, &[&a, &b], &ndarray(&device, &[1, 1])),
        Err(Error::DimensionMismatch {
            field: "destination dimension",
            expected: 4,
            actual: 1,
        })
    );
    let with_addend = NDArrayMatrixMultiplication::new(&device, 3).expect("kernel");
    assert_eq!(
        with_addend.encode_to_destination(
            &command_buffer,
            &[&a, &b, &ndarray(&device, &[3, 2])],
            &ndarray(&device, &[4, 2])
        ),
        Err(Error::DimensionMismatch {
            field: "addend dimension",
            expected: 4,
            actual: 3,
        })
    );
    let result = kernel
        .encode(
            &command_buffer,
            &[&ndarray(&device, &[3, 2, 1]), &ndarray(&device, &[4, 3, 5])],
        )
        .expect("broadcast batch");
    assert_eq!(result.length_of_dimension(0), 4);
    assert_eq!(result.length_of_dimension(1), 2);
    assert_eq!(result.length_of_dimension(2), 5);
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
}

#[test]
fn matrix_multiplication_checks_m_n_k_and_data_types() {
    let device = device();
    let kernel = MatrixMultiplication::new_simple(&device, 4, 4, 4).expect("kernel");
    let queue = device.new_command_queue().expect("queue");
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let (_a, two) = matrix(&device, 2, 2, data_type::FLOAT32);
    let (_b, four) = matrix(&device, 4, 4, data_type::FLOAT32);
    assert_eq!(
        kernel.encode(&command_buffer, &two, &four, &four),
        Err(Error::DimensionMismatch {
            field: "left rows",
            expected: 4,
            actual: 2,
        })
    );
    assert_eq!(
        kernel.encode(&command_buffer, &four, &two, &four),
        Err(Error::DimensionMismatch {
            field: "right rows",
            expected: 4,
            actual: 2,
        })
    );
    assert_eq!(
        kernel.encode(&command_buffer, &four, &four, &two),
        Err(Error::DimensionMismatch {
            field: "result rows",
            expected: 4,
            actual: 2,
        })
    );
    let (_c, wide) = matrix(&device, 4, 2, data_type::FLOAT32);
    assert_eq!(
        kernel.encode(&command_buffer, &wide, &four, &four),
        Err(Error::DimensionMismatch {
            field: "left columns",
            expected: 4,
            actual: 2,
        })
    );
    let (_d, int) = matrix(&device, 4, 4, data_type::INT32);
    assert!(matches!(
        kernel.encode(&command_buffer, &int, &int, &int),
        Err(Error::InvalidArgument(_))
    ));
    let (_e, half) = matrix(&device, 4, 4, data_type::FLOAT16);
    assert!(matches!(
        kernel.encode(&command_buffer, &half, &four, &four),
        Err(Error::InvalidArgument(_))
    ));
    kernel
        .encode(&command_buffer, &four, &half, &four)
        .expect("f32 x f16 -> f32 is supported");
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");

    let transposed = MatrixMultiplication::new(
        &device,
        MatrixMultiplicationDescriptor::with_options(true, false, 2, 4, 3, 1.0, 0.0),
    )
    .expect("transposed kernel");
    assert!(transposed.descriptor().transpose_left);
    let left_values: Vec<f32> = (1_u8..=6).map(f32::from).collect();
    let right_values: Vec<f32> = (1_u8..=12).map(f32::from).collect();
    let left_buffer = filled(&device, &left_values, 24);
    let right_buffer = filled(&device, &right_values, 48);
    let result_buffer = buffer(&device, 32);
    let left = Matrix::new_with_buffer(
        &left_buffer,
        MatrixDescriptor::contiguous(3, 2, data_type::FLOAT32).expect("desc"),
    )
    .expect("left");
    let right = Matrix::new_with_buffer(
        &right_buffer,
        MatrixDescriptor::contiguous(3, 4, data_type::FLOAT32).expect("desc"),
    )
    .expect("right");
    let result = Matrix::new_with_buffer(
        &result_buffer,
        MatrixDescriptor::contiguous(2, 4, data_type::FLOAT32).expect("desc"),
    )
    .expect("result");
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    assert_eq!(
        transposed.encode(&command_buffer, &result, &right, &result),
        Err(Error::DimensionMismatch {
            field: "left rows",
            expected: 3,
            actual: 2,
        })
    );
    transposed
        .encode(&command_buffer, &left, &right, &result)
        .expect("transposed left is stored as K x M");
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
    assert_eq!(
        read(&result_buffer, 8),
        vec![61.0, 70.0, 79.0, 88.0, 76.0, 88.0, 100.0, 112.0]
    );
}
