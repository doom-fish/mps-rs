use apple_metal::{resource_options, MetalBuffer, MetalDevice};
use apple_mps::{
    feature_channel_format, image_layout, state_resource_type, Error, HistogramInfo, Image,
    ImageDescriptor, ImageHistogram, ImageReadWriteParams, ImageRegion, Predicate, State,
    StateTextureInfo,
};

fn device() -> MetalDevice {
    MetalDevice::system_default().expect("no Metal device available")
}

fn buffer(device: &MetalDevice, length: usize) -> MetalBuffer {
    device
        .new_buffer(length, resource_options::STORAGE_MODE_SHARED)
        .expect("buffer")
}

const fn histogram_info(entries: usize) -> HistogramInfo {
    HistogramInfo {
        number_of_entries: entries,
        histogram_for_alpha: false,
        min_pixel_value: [0.0; 4],
        max_pixel_value: [1.0; 4],
    }
}

#[test]
fn state_indices_past_the_resources_read_as_empty() {
    let device = device();
    let state = State::new_with_buffer_size(&device, 16).expect("state");
    assert_eq!(state.resource_count(), 1);
    assert_eq!(state.buffer_size_at_index(0), 16);
    assert_eq!(state.buffer_size_at_index(5), 0);
    assert_eq!(state.resource_type_at_index(5), state_resource_type::NONE);
    assert_eq!(state.texture_info_at_index(5), StateTextureInfo::default());
}

#[test]
fn predicates_must_fit_in_their_buffer() {
    let device = device();
    let storage = buffer(&device, 8);
    assert!(Predicate::new_with_buffer(&storage, 8).is_none());
    assert!(Predicate::new_with_buffer(&storage, 64).is_none());
    assert!(Predicate::new_with_buffer(&storage, 2).is_none());
    assert!(Predicate::new_with_buffer(&storage, usize::MAX).is_none());
    let predicate = Predicate::new_with_buffer(&storage, 4).expect("last word");
    assert_eq!(predicate.predicate_offset(), 4);
}

#[test]
fn histograms_need_room_for_every_bin() {
    let device = device();
    assert!(ImageHistogram::new(&device, histogram_info(3)).is_none());
    assert!(ImageHistogram::new(&device, histogram_info(0)).is_none());
    let histogram = ImageHistogram::new(&device, histogram_info(256)).expect("histogram");
    let image = Image::new(
        &device,
        ImageDescriptor::new(4, 4, 4, feature_channel_format::UNORM8),
    )
    .expect("image");
    image
        .write_bytes(
            &[0_u8; 64],
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            16,
            ImageRegion::whole(4, 4),
            ImageReadWriteParams::all(4),
            0,
        )
        .expect("write image");
    let required = histogram.histogram_size_for_source_format(image.pixel_format());
    assert_eq!(required, 3 * 256 * 4);
    let queue = device.new_command_queue().expect("queue");
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    assert_eq!(
        histogram.encode_image(&command_buffer, &image, &buffer(&device, 16), 0),
        Err(Error::BufferTooSmall {
            required,
            length: 16,
        })
    );
    let output = buffer(&device, required + 32);
    assert_eq!(
        histogram.encode_image(&command_buffer, &image, &output, 4),
        Err(Error::Misaligned {
            field: "histogram_offset",
            value: 4,
            alignment: 32,
        })
    );
    assert_eq!(
        histogram.encode_image(&command_buffer, &image, &output, 64),
        Err(Error::BufferTooSmall {
            required: required + 64,
            length: required + 32,
        })
    );
    histogram
        .encode_image(&command_buffer, &image, &output, 32)
        .expect("fits at offset 32");
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
    let mut bins = vec![0_u8; required];
    unsafe { output.read_bytes(32, &mut bins) }.expect("read bins");
    let red: u32 = bins[..1024]
        .chunks_exact(4)
        .map(|bin| u32::from_ne_bytes([bin[0], bin[1], bin[2], bin[3]]))
        .sum();
    assert_eq!(red, 16);
}

trait AmbiguousIfSync<A> {
    fn check() {}
}

impl<T: ?Sized> AmbiguousIfSync<()> for T {}

impl<T: ?Sized + Sync> AmbiguousIfSync<u8> for T {}

const fn assert_send<T: Send>() {}

const fn assert_sync<T: Sync>() {}

#[test]
fn kernels_and_descriptors_are_send_but_not_sync() {
    assert_send::<apple_mps::MatrixMultiplication>();
    assert_send::<apple_mps::ImageGaussianBlur>();
    assert_send::<apple_mps::CnnConvolutionDescriptor>();
    assert_send::<apple_mps::PolygonAccelerationStructure>();
    <apple_mps::MatrixMultiplication as AmbiguousIfSync<_>>::check();
    <apple_mps::NDArrayMatrixMultiplication as AmbiguousIfSync<_>>::check();
    <apple_mps::NDArrayDescriptor as AmbiguousIfSync<_>>::check();
    <apple_mps::ImageGaussianBlur as AmbiguousIfSync<_>>::check();
    <apple_mps::ImageHistogram as AmbiguousIfSync<_>>::check();
    <apple_mps::CnnConvolution as AmbiguousIfSync<_>>::check();
    <apple_mps::CnnConvolutionDescriptor as AmbiguousIfSync<_>>::check();
    <apple_mps::NNGraph as AmbiguousIfSync<_>>::check();
    <apple_mps::NNOptimizerAdam as AmbiguousIfSync<_>>::check();
    <apple_mps::RayIntersector as AmbiguousIfSync<_>>::check();
    <apple_mps::PolygonAccelerationStructure as AmbiguousIfSync<_>>::check();
    <apple_mps::MpsCommandBuffer as AmbiguousIfSync<_>>::check();
    <apple_mps::State as AmbiguousIfSync<_>>::check();
    assert_sync::<apple_mps::Matrix>();
    assert_sync::<apple_mps::Vector>();
    assert_sync::<apple_mps::NDArray>();
    assert_sync::<Image>();
    assert_sync::<Predicate>();
}
