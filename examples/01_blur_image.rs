use apple_metal::MetalDevice;
use apple_mps::{feature_channel_format, Image, ImageDescriptor, ImageGaussianBlur};

fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device
        .new_command_queue()
        .expect("failed to create command queue");

    let descriptor = ImageDescriptor::new(256, 256, 1, feature_channel_format::FLOAT32);
    let src = Image::new(&device, descriptor).expect("failed to allocate source image");
    let dst = Image::new(&device, descriptor).expect("failed to allocate destination image");

    let mut impulse = vec![0.0_f32; 256 * 256];
    let center_index = (256 / 2) * 256 + (256 / 2);
    impulse[center_index] = 1.0;
    src.write_f32(&impulse)
        .expect("failed to upload source image");

    let blur = ImageGaussianBlur::new(&device, 2.0).expect("failed to create gaussian blur");
    let command_buffer = queue
        .new_command_buffer()
        .expect("failed to allocate command buffer");
    blur.encode_image(&command_buffer, &src, &dst);
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let output = dst.read_f32().expect("failed to download blurred image");
    let center_value = output[center_index];
    let neighbor_value = output[center_index + 1];
    let corner_value = output[0];
    let total_energy: f32 = output.iter().sum();

    assert!(
        center_value < 1.0,
        "center should have blurred away from impulse"
    );
    assert!(
        neighbor_value > 0.0,
        "neighbor should receive energy after blur"
    );
    assert!(
        center_value > neighbor_value,
        "center should remain strongest sample"
    );
    assert!(corner_value.abs() < 1.0e-6, "corners should stay at zero");
    assert!(
        (total_energy - 1.0).abs() < 1.0e-2,
        "gaussian blur should preserve energy, got {total_energy}"
    );

    println!(
        "blur smoke passed: center={center_value:.6} neighbor={neighbor_value:.6} sum={total_energy:.6}"
    );
}
