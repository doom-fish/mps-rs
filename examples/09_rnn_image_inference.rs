use apple_metal::MetalDevice;
use apple_mps::{feature_channel_format, Image, ImageDescriptor, RnnImageInferenceLayer, RnnSingleGateDescriptor};

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device.new_command_queue().expect("command queue");

    let single_gate = RnnSingleGateDescriptor::new(1, 1).expect("single gate descriptor");
    single_gate.set_use_layer_input_unit_transform_mode(true);
    let descriptor = single_gate.as_descriptor().expect("base descriptor");
    let layer = RnnImageInferenceLayer::new(&device, &descriptor).expect("rnn layer");

    let image_descriptor = ImageDescriptor::new(1, 1, 1, feature_channel_format::FLOAT32);
    let src0 = Image::new(&device, image_descriptor).expect("src0");
    let src1 = Image::new(&device, image_descriptor).expect("src1");
    let dst0 = Image::new(&device, image_descriptor).expect("dst0");
    let dst1 = Image::new(&device, image_descriptor).expect("dst1");
    src0.write_f32(&[0.25]).expect("write src0");
    src1.write_f32(&[0.75]).expect("write src1");

    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let recurrent_state = layer
        .encode_sequence(&command_buffer, &[&src0, &src1], &[&dst0, &dst1], None)
        .expect("recurrent state");
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let recurrent_output = recurrent_state
        .recurrent_output_image_for_layer_index(0)
        .expect("recurrent output image");
    println!(
        "{:?} {}x{}x{}",
        dst1.read_f32().expect("dst1 output"),
        recurrent_output.width(),
        recurrent_output.height(),
        recurrent_output.feature_channels()
    );
}
