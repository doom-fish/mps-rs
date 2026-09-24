use apple_metal::{CommandBufferError, MetalDevice};
use apple_mps::{
    data_type, feature_channel_format, state_batch_increment_read_count, state_batch_synchronize,
    CnnNeuronReluNode, Error, Image, ImageDescriptor, NDArray, NDArrayDescriptor,
    NDArrayMatrixMultiplication, NNGraph, NNImageNode, RnnImageInferenceLayer,
    RnnSingleGateDescriptor, State,
};
use std::process::Command;

const VALIDATION_CHILD: &str = "APPLE_MPS_VALIDATION_CHILD";

fn device() -> MetalDevice {
    MetalDevice::system_default().expect("no Metal device available")
}

#[test]
fn read_counts_refuse_what_mps_aborts_on() {
    let device = device();
    let queue = device.new_command_queue().expect("command queue");
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let temporary =
        State::temporary_with_buffer_size(&command_buffer, 64).expect("temporary state");
    let persistent = State::new_with_buffer_size(&device, 64).expect("persistent state");
    assert!(temporary.is_temporary());
    assert!(!persistent.is_temporary());

    assert!(matches!(
        persistent.set_read_count(2),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        state_batch_increment_read_count(&[&temporary], -3),
        Err(Error::InvalidArgument(_))
    ));
    assert_eq!(temporary.read_count(), 1);
    assert_eq!(
        state_batch_increment_read_count(&[&temporary, &temporary, &persistent], 2)
            .expect("raise the read count"),
        2
    );
    assert_eq!(temporary.read_count(), 3);
    assert!(matches!(
        temporary.synchronize_on_command_buffer(&command_buffer),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        state_batch_synchronize(&[&persistent, &temporary], &command_buffer),
        Err(Error::InvalidArgument(_))
    ));
    temporary.set_read_count(0).expect("return the storage");
    assert!(matches!(
        temporary.set_read_count(2),
        Err(Error::InvalidArgument(_))
    ));
    temporary.set_read_count(0).expect("zero again");
    assert!(matches!(
        state_batch_increment_read_count(&[&temporary], -1),
        Err(Error::InvalidArgument(_))
    ));
    state_batch_synchronize(&[&persistent], &command_buffer).expect("persistent synchronize");
    persistent
        .synchronize_on_command_buffer(&command_buffer)
        .expect("persistent synchronize");
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
}

#[test]
fn read_counts_wait_for_open_encoders() {
    let device = device();
    let queue = device.new_command_queue().expect("command queue");
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let temporary =
        State::temporary_with_buffer_size(&command_buffer, 64).expect("temporary state");
    let leaked = State::temporary_with_buffer_size(&command_buffer, 64).expect("temporary state");
    let encoder = command_buffer
        .new_compute_command_encoder()
        .expect("compute encoder");
    assert!(matches!(
        temporary.set_read_count(0),
        Err(Error::CommandBuffer(CommandBufferError::ActiveEncoder))
    ));
    assert!(matches!(
        state_batch_increment_read_count(&[&temporary], 1),
        Err(Error::CommandBuffer(CommandBufferError::ActiveEncoder))
    ));
    assert_eq!(temporary.read_count(), 1);
    drop(leaked);
    encoder.end_encoding().expect("end encoding");
    temporary.set_read_count(2).expect("raise the read count");
    command_buffer.commit().expect("commit");
    assert_eq!(
        state_batch_increment_read_count(&[&temporary], -1).expect("lower after commit"),
        1
    );
    temporary.set_read_count(0).expect("zero after commit");
    command_buffer.wait_until_completed().expect("completed");
}

#[test]
fn temporary_recurrent_outputs_are_refused() {
    let device = device();
    let descriptor = RnnSingleGateDescriptor::new(1, 1)
        .expect("single gate descriptor")
        .as_descriptor()
        .expect("descriptor");
    let layer = RnnImageInferenceLayer::new(&device, &descriptor).expect("rnn layer");
    assert!(matches!(
        layer.set_recurrent_output_is_temporary(true),
        Err(Error::Unsupported(_))
    ));
    assert!(!layer.recurrent_output_is_temporary());
    layer
        .set_recurrent_output_is_temporary(false)
        .expect("persistent recurrent output");
}

#[test]
fn temporary_objects_satisfy_the_validation_layer() {
    if std::env::var_os(VALIDATION_CHILD).is_some() {
        exercise_temporary_objects();
        return;
    }
    let output = Command::new(std::env::current_exe().expect("test binary"))
        .args([
            "--exact",
            "temporary_objects_satisfy_the_validation_layer",
            "--test-threads=1",
        ])
        .env(VALIDATION_CHILD, "1")
        .env("MTL_DEBUG_LAYER", "1")
        .output()
        .expect("rerun the test under the Metal validation layer");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(output.status.success(), "{stderr}");
    assert!(stderr.contains("Metal API Validation Enabled"), "{stderr}");
}

fn exercise_temporary_objects() {
    let device = device();
    let queue = device.new_command_queue().expect("command queue");
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let raised = State::temporary_with_buffer_size(&command_buffer, 32).expect("temporary state");
    state_batch_increment_read_count(&[&raised], 4).expect("raise the read count");
    drop(raised);
    let kept = State::temporary_with_buffer_size(&command_buffer, 16).expect("temporary state");
    kept.set_read_count(7).expect("raise the read count");
    let leaked = State::temporary_with_buffer_size(&command_buffer, 8).expect("temporary state");
    let encoder = command_buffer
        .new_compute_command_encoder()
        .expect("compute encoder");
    drop(leaked);
    encoder.end_encoding().expect("end encoding");

    let descriptor = NDArrayDescriptor::with_dimension_sizes(data_type::FLOAT32, &[2, 2, 1, 1])
        .expect("descriptor");
    let array = NDArray::new(&device, &descriptor).expect("array");
    let multiply = NDArrayMatrixMultiplication::new(&device, 2).expect("matrix multiplication");
    let product = multiply
        .encode(&command_buffer, &[&array, &array])
        .expect("product");
    let first = multiply
        .encode(&command_buffer, &[&product, &array])
        .expect("first reuse");
    let second = multiply
        .encode(&command_buffer, &[&product, &array])
        .expect("second reuse");

    let source_node = NNImageNode::new().expect("source node");
    source_node.set_format(feature_channel_format::FLOAT32);
    let first_relu = CnnNeuronReluNode::new(&source_node, 0.0).expect("relu node");
    let exported = first_relu.result_image().expect("intermediate node");
    exported.set_export_from_graph(true);
    let second_relu = CnnNeuronReluNode::new(&exported, 0.0).expect("relu node");
    let result_node = second_relu.result_image().expect("result node");
    let graph = NNGraph::new(&device, &result_node, true).expect("graph");
    graph.set_format(feature_channel_format::FLOAT32);
    let source = Image::new(
        &device,
        ImageDescriptor::new(2, 2, 1, feature_channel_format::FLOAT32),
    )
    .expect("source image");
    source
        .write_f32(&[-1.0, 0.5, 2.0, -0.25])
        .expect("write source image");
    let graph_results = [
        graph
            .encode(&command_buffer, &[&source])
            .expect("graph encode"),
        graph
            .encode(&command_buffer, &[&source])
            .expect("graph encode"),
    ];

    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
    drop((kept, product, first, second));
    for result in &graph_results {
        let values = result.read_f32().expect("read graph result");
        assert!(
            values
                .iter()
                .zip([0.0_f32, 0.5, 2.0, 0.0])
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-6),
            "{values:?}"
        );
    }
}
