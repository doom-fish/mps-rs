use apple_metal::MetalDevice;
use apple_mps::{
    cnn_accumulator_precision_option, cnn_convolution_flags, feature_channel_format,
    CnnConvolution, CnnConvolutionDescriptor, Image, ImageDescriptor,
};

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device.new_command_queue().expect("command queue");

    let descriptor = CnnConvolutionDescriptor::new(1, 1, 1, 1).expect("descriptor");
    let convolution = CnnConvolution::new(
        &device,
        &descriptor,
        &[2.0],
        Some(&[0.5]),
        cnn_convolution_flags::NONE,
    )
    .expect("convolution");
    convolution.set_accumulator_precision_option(cnn_accumulator_precision_option::FLOAT);

    let image_descriptor = ImageDescriptor::new(2, 2, 1, feature_channel_format::FLOAT32);
    let source = Image::new(&device, image_descriptor).expect("source image");
    let destination = Image::new(&device, image_descriptor).expect("destination image");
    source.write_f32(&[1.0, 2.0, 3.0, 4.0]).expect("write source");

    let command_buffer = queue.new_command_buffer().expect("command buffer");
    convolution.encode_image(&command_buffer, &source, &destination);
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let output = destination.read_f32().expect("output");
    println!("{output:?}");
}
