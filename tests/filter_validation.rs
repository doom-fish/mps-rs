use apple_metal::{
    pixel_format, storage_mode, texture_type, texture_usage, CommandBuffer, CommandQueue,
    MetalDevice, MetalTexture, TextureDescriptor,
};
use apple_mps::{
    feature_channel_format, hint_temporary_memory_high_water_mark, set_heap_cache_duration, Error,
    HistogramInfo, Image, ImageAdd, ImageBox, ImageConvolution, ImageDescriptor, ImageGaussianBlur,
    ImageHistogram, ImageMedian, ImageReduceRowSum, ImageRegion, ImageSobel, ImageStatisticsMean,
    ImageStatisticsMinAndMax, ImageThresholdBinary, MpsCommandBuffer, State,
};

fn device() -> MetalDevice {
    MetalDevice::system_default().expect("Metal device")
}

fn queue(device: &MetalDevice) -> CommandQueue {
    device.new_command_queue().expect("command queue")
}

fn texture_with(device: &MetalDevice, descriptor: TextureDescriptor) -> MetalTexture {
    device.new_texture(descriptor).expect("texture")
}

fn texture(device: &MetalDevice, format: usize, width: usize, height: usize) -> MetalTexture {
    texture_with(device, TextureDescriptor::new_2d(width, height, format))
}

fn array_texture(device: &MetalDevice, format: usize, layers: usize) -> MetalTexture {
    texture_with(
        device,
        TextureDescriptor {
            texture_type: texture_type::TYPE_2D_ARRAY,
            array_length: layers,
            ..TextureDescriptor::new_2d(8, 8, format)
        },
    )
}

fn image(device: &MetalDevice, feature_channels: usize) -> Image {
    Image::new(
        device,
        ImageDescriptor::new(8, 8, feature_channels, feature_channel_format::FLOAT32),
    )
    .expect("image")
}

fn run(command_buffer: &CommandBuffer) {
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
}

fn committed(queue: &CommandQueue) -> CommandBuffer {
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    run(&command_buffer);
    command_buffer
}

const fn unsupported(operand: &'static str, pixel_format: usize) -> Error {
    Error::UnsupportedPixelFormat {
        operand,
        pixel_format,
    }
}

#[test]
fn integer_formats_are_rejected() {
    let device = device();
    let queue = queue(&device);
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let blur = ImageGaussianBlur::new(&device, 1.0).expect("blur");
    let integer = texture(&device, pixel_format::RGBA8UINT, 8, 8);
    let normalized = texture(&device, pixel_format::RGBA8UNORM, 8, 8);
    assert_eq!(
        blur.encode_texture(&command_buffer, &integer, &normalized),
        Err(unsupported("source", pixel_format::RGBA8UINT))
    );
    assert_eq!(
        blur.encode_texture(&command_buffer, &normalized, &integer),
        Err(unsupported("destination", pixel_format::RGBA8UINT))
    );
    let mono_integer = texture(&device, pixel_format::R32UINT, 8, 8);
    let threshold = ImageThresholdBinary::new(&device, 0.5, 1.0).expect("threshold");
    assert_eq!(
        threshold.encode_texture(&command_buffer, &normalized, &mono_integer),
        Err(unsupported("destination", pixel_format::R32UINT))
    );
    let add = ImageAdd::new(&device).expect("add");
    assert_eq!(
        add.encode_texture(&command_buffer, &integer, &normalized, &normalized),
        Err(unsupported("primary", pixel_format::RGBA8UINT))
    );
    let other = texture(&device, pixel_format::RGBA8UNORM, 8, 8);
    assert_eq!(
        add.encode_texture(&command_buffer, &normalized, &integer, &other),
        Err(unsupported("secondary", pixel_format::RGBA8UINT))
    );
}

#[test]
fn destinations_must_be_writable() {
    let device = device();
    let queue = queue(&device);
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let blur = ImageGaussianBlur::new(&device, 1.0).expect("blur");
    let source = texture(&device, pixel_format::RGBA8UNORM, 8, 8);
    let depth = texture_with(
        &device,
        TextureDescriptor {
            storage_mode: storage_mode::PRIVATE,
            ..TextureDescriptor::new_2d(8, 8, pixel_format::DEPTH32FLOAT)
        },
    );
    assert_eq!(
        blur.encode_texture(&command_buffer, &source, &depth),
        Err(unsupported("destination", pixel_format::DEPTH32FLOAT))
    );
    let packed = texture(&device, pixel_format::ABGR4UNORM, 8, 8);
    assert_eq!(
        blur.encode_texture(&command_buffer, &source, &packed),
        Err(unsupported("destination", pixel_format::ABGR4UNORM))
    );
    let read_only = texture_with(
        &device,
        TextureDescriptor {
            usage: texture_usage::SHADER_READ,
            ..TextureDescriptor::new_2d(8, 8, pixel_format::RGBA8UNORM)
        },
    );
    assert!(matches!(
        blur.encode_texture(&command_buffer, &source, &read_only),
        Err(Error::InvalidArgument(_))
    ));
    let write_only = texture_with(
        &device,
        TextureDescriptor {
            usage: texture_usage::SHADER_WRITE,
            ..TextureDescriptor::new_2d(8, 8, pixel_format::RGBA8UNORM)
        },
    );
    let destination = texture(&device, pixel_format::RGBA8UNORM, 8, 8);
    assert!(matches!(
        blur.encode_texture(&command_buffer, &write_only, &destination),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn color_models_follow_each_kernel_group() {
    let device = device();
    let queue = queue(&device);
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let rgba = texture(&device, pixel_format::RGBA8UNORM, 8, 8);
    let rgb = texture(&device, pixel_format::RG11B10FLOAT, 8, 8);
    let rg = texture(&device, pixel_format::RG8UNORM, 8, 8);
    let mono = texture(&device, pixel_format::R8UNORM, 8, 8);
    let blur = ImageGaussianBlur::new(&device, 1.0).expect("blur");
    assert!(matches!(
        blur.encode_texture(&command_buffer, &rgba, &mono),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        blur.encode_texture(&command_buffer, &mono, &rg),
        Err(Error::InvalidArgument(_))
    ));
    let median = ImageMedian::new(&device, 3).expect("median");
    assert!(matches!(
        median.encode_texture(&command_buffer, &rg, &rgba),
        Err(Error::InvalidArgument(_))
    ));
    let reduce = ImageReduceRowSum::new(&device).expect("reduce");
    assert!(matches!(
        reduce.encode_texture(&command_buffer, &mono, &rgba),
        Err(Error::InvalidArgument(_))
    ));
    let sobel = ImageSobel::new(&device).expect("sobel");
    assert!(matches!(
        sobel.encode_texture(&command_buffer, &rgba, &rg),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        sobel.encode_texture(&command_buffer, &rgba, &rgb),
        Err(Error::InvalidArgument(_))
    ));
    blur.encode_texture(&command_buffer, &rgba, &rgb)
        .expect("RGB color models match");
    sobel
        .encode_texture(&command_buffer, &rgba, &mono)
        .expect("Sobel converts to a single channel");
    let threshold = ImageThresholdBinary::new(&device, 0.5, 1.0).expect("threshold");
    threshold
        .encode_texture(&command_buffer, &rg, &rgba)
        .expect("threshold accepts any channel count");
    run(&command_buffer);
}

#[test]
fn statistics_need_matching_channels_and_room() {
    let device = device();
    let queue = queue(&device);
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let source = texture(&device, pixel_format::RGBA16FLOAT, 8, 8);
    let min_max = ImageStatisticsMinAndMax::new(&device).expect("min and max");
    let mono = texture(&device, pixel_format::R16FLOAT, 2, 1);
    assert!(matches!(
        min_max.encode_texture(&command_buffer, &source, &mono),
        Err(Error::InvalidArgument(_))
    ));
    let rgb = texture(&device, pixel_format::RG11B10FLOAT, 2, 1);
    assert!(matches!(
        min_max.encode_texture(&command_buffer, &source, &rgb),
        Err(Error::InvalidArgument(_))
    ));
    let narrow = texture(&device, pixel_format::RGBA16FLOAT, 1, 4);
    assert_eq!(
        min_max.encode_texture(&command_buffer, &source, &narrow),
        Err(Error::DimensionMismatch {
            field: "clipped destination width",
            expected: 2,
            actual: 1,
        })
    );
    let wide = texture(&device, pixel_format::RGBA16FLOAT, 8, 1);
    min_max.set_clip_rect(ImageRegion::new(7, 0, 0, 4, 1, 1));
    assert!(matches!(
        min_max.encode_texture(&command_buffer, &source, &wide),
        Err(Error::DimensionMismatch { actual: 1, .. })
    ));
    min_max.set_clip_rect(ImageRegion::new(100, 0, 0, 4, 1, 1));
    assert!(matches!(
        min_max.encode_texture(&command_buffer, &source, &wide),
        Err(Error::DimensionMismatch { actual: 0, .. })
    ));
    min_max.set_clip_rect(ImageRegion::new(6, 0, 0, 2, 1, 1));
    min_max
        .encode_texture(&command_buffer, &source, &wide)
        .expect("two pixels fit");
    let mean = ImageStatisticsMean::new(&device).expect("mean");
    let single = texture(&device, pixel_format::RGBA16FLOAT, 1, 1);
    mean.encode_texture(&command_buffer, &source, &single)
        .expect("the mean needs one pixel");
    let alpha = texture(&device, pixel_format::A8UNORM, 8, 8);
    let alpha_destination = texture(&device, pixel_format::A8UNORM, 1, 1);
    assert_eq!(
        mean.encode_texture(&command_buffer, &alpha, &alpha_destination),
        Err(unsupported("source", pixel_format::A8UNORM))
    );
    run(&command_buffer);
}

#[test]
fn alpha_formats_follow_kernel_support() {
    let device = device();
    let queue = queue(&device);
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let source = texture(&device, pixel_format::A8UNORM, 8, 8);
    let destination = texture(&device, pixel_format::A8UNORM, 8, 8);
    let median = ImageMedian::new(&device, 3).expect("median");
    assert_eq!(
        median.encode_texture(&command_buffer, &source, &destination),
        Err(unsupported("source", pixel_format::A8UNORM))
    );
    let mono = texture(&device, pixel_format::R8UNORM, 8, 8);
    let blur = ImageGaussianBlur::new(&device, 1.0).expect("blur");
    assert!(matches!(
        blur.encode_texture(&command_buffer, &source, &mono),
        Err(Error::InvalidArgument(_))
    ));
    blur.encode_texture(&command_buffer, &source, &destination)
        .expect("alpha to alpha");
    run(&command_buffer);
}

#[test]
fn kernels_do_not_run_in_place() {
    let device = device();
    let queue = queue(&device);
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let blur = ImageGaussianBlur::new(&device, 1.0).expect("blur");
    let target = texture(&device, pixel_format::RGBA8UNORM, 8, 8);
    assert!(matches!(
        blur.encode_texture(&command_buffer, &target, &target),
        Err(Error::InvalidArgument(_))
    ));
    let view = target.new_view(pixel_format::RGBA8UNORM).expect("view");
    assert_eq!(
        blur.encode_texture(&command_buffer, &target, &view),
        Err(Error::Rejected("MPSUnaryImageKernel encode"))
    );
    let rgba = image(&device, 4);
    assert!(matches!(
        blur.encode_image(&command_buffer, &rgba, &rgba),
        Err(Error::InvalidArgument(_))
    ));
    let first = Image::from_texture(&target, 4).expect("first image");
    let second = Image::from_texture(&target, 4).expect("second image");
    assert!(matches!(
        blur.encode_image(&command_buffer, &first, &second),
        Err(Error::InvalidArgument(_))
    ));
    let add = ImageAdd::new(&device).expect("add");
    let primary = texture(&device, pixel_format::RGBA8UNORM, 8, 8);
    assert!(matches!(
        add.encode_texture(&command_buffer, &primary, &target, &target),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        add.encode_texture(&command_buffer, &primary, &target, &primary),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn texture_types_must_match() {
    let device = device();
    let queue = queue(&device);
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let blur = ImageGaussianBlur::new(&device, 1.0).expect("blur");
    let flat = texture(&device, pixel_format::RGBA8UNORM, 8, 8);
    let layered = array_texture(&device, pixel_format::RGBA8UNORM, 2);
    assert!(matches!(
        blur.encode_texture(&command_buffer, &flat, &layered),
        Err(Error::InvalidArgument(_))
    ));
    let three = array_texture(&device, pixel_format::RGBA8UNORM, 3);
    assert_eq!(
        blur.encode_texture(&command_buffer, &three, &layered),
        Err(Error::DimensionMismatch {
            field: "destination array_length",
            expected: 3,
            actual: 2,
        })
    );
    let volume = |depth| {
        texture_with(
            &device,
            TextureDescriptor {
                texture_type: texture_type::TYPE_3D,
                depth,
                ..TextureDescriptor::new_2d(8, 8, pixel_format::RGBA8UNORM)
            },
        )
    };
    assert!(matches!(
        blur.encode_texture(&command_buffer, &volume(2), &volume(2)),
        Err(Error::InvalidArgument(_))
    ));
    let add = ImageAdd::new(&device).expect("add");
    let other = texture(&device, pixel_format::RGBA8UNORM, 8, 8);
    assert!(matches!(
        add.encode_texture(&command_buffer, &flat, &layered, &other),
        Err(Error::InvalidArgument(_))
    ));
    let other_layers = array_texture(&device, pixel_format::RGBA8UNORM, 2);
    let layered_destination = array_texture(&device, pixel_format::RGBA8UNORM, 2);
    blur.encode_texture(&command_buffer, &other_layers, &layered_destination)
        .expect("matching 2D arrays");
    run(&command_buffer);
}

#[test]
fn unary_kernels_take_at_most_four_feature_channels() {
    let device = device();
    let queue = queue(&device);
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let blur = ImageGaussianBlur::new(&device, 1.0).expect("blur");
    let wide = image(&device, 8);
    let wide_destination = image(&device, 8);
    assert!(matches!(
        blur.encode_image(&command_buffer, &wide, &wide_destination),
        Err(Error::InvalidArgument(_))
    ));
    let add = ImageAdd::new(&device).expect("add");
    let secondary = image(&device, 8);
    add.encode_image(&command_buffer, &wide, &secondary, &wide_destination)
        .expect("binary kernels accept wide images");
    let narrow = image(&device, 4);
    assert!(matches!(
        add.encode_image(&command_buffer, &narrow, &secondary, &wide_destination),
        Err(Error::InvalidArgument(_))
    ));
    run(&command_buffer);
}

#[test]
fn committed_command_buffers_are_rejected() {
    let device = device();
    let queue = queue(&device);
    let done = committed(&queue);
    let blur = ImageGaussianBlur::new(&device, 1.0).expect("blur");
    let source = texture(&device, pixel_format::RGBA8UNORM, 8, 8);
    let destination = texture(&device, pixel_format::RGBA8UNORM, 8, 8);
    assert!(matches!(
        blur.encode_texture(&done, &source, &destination),
        Err(Error::NotRecording { .. })
    ));
    let add = ImageAdd::new(&device).expect("add");
    let other = texture(&device, pixel_format::RGBA8UNORM, 8, 8);
    assert!(matches!(
        add.encode_texture(&done, &source, &other, &destination),
        Err(Error::NotRecording { .. })
    ));
    let histogram = ImageHistogram::new(
        &device,
        HistogramInfo {
            number_of_entries: 256,
            histogram_for_alpha: false,
            min_pixel_value: [0.0; 4],
            max_pixel_value: [1.0; 4],
        },
    )
    .expect("histogram");
    let bins = device
        .new_buffer(4096, apple_metal::resource_options::STORAGE_MODE_SHARED)
        .expect("bins");
    assert!(matches!(
        histogram.encode_texture(&done, &source, &bins, 0),
        Err(Error::NotRecording { .. })
    ));
    assert!(matches!(
        hint_temporary_memory_high_water_mark(&done, 1 << 20),
        Err(Error::NotRecording { .. })
    ));
    assert!(matches!(
        set_heap_cache_duration(&done, 1.0),
        Err(Error::NotRecording { .. })
    ));
    assert!(State::temporary(&done).is_none());
    assert!(State::temporary_with_buffer_size(&done, 64).is_none());
    let wrapped = MpsCommandBuffer::new_with_command_buffer(&done).expect("MPS command buffer");
    assert_eq!(
        wrapped.prefetch_heap_for_workload_size(1 << 20),
        Err(Error::Rejected("MPSCommandBuffer heap prefetch"))
    );
    let live = queue.new_command_buffer().expect("command buffer");
    hint_temporary_memory_high_water_mark(&live, 1 << 20).expect("hint");
    set_heap_cache_duration(&live, 1.0).expect("heap cache duration");
    let prefetching = MpsCommandBuffer::new_with_command_buffer(&live).expect("MPS command buffer");
    prefetching
        .prefetch_heap_for_workload_size(1 << 16)
        .expect("prefetch");
    blur.encode_texture(&live, &source, &destination)
        .expect("recording buffer");
    run(&live);
}

#[test]
fn kernel_sizes_are_checked() {
    let device = device();
    assert!(ImageBox::new(&device, 2, 3).is_none());
    assert!(ImageBox::new(&device, 3, 4).is_none());
    assert!(ImageBox::new(&device, 0, 1).is_none());
    assert!(ImageBox::new(&device, 3, 5).is_some());
    assert!(ImageConvolution::new(&device, 2, 3, &[0.0; 6]).is_none());
    assert!(ImageConvolution::new(&device, 3, 2, &[0.0; 6]).is_none());
    assert!(ImageConvolution::new(&device, 0, 0, &[]).is_none());
    assert!(ImageConvolution::new(&device, 3, 1, &[0.25, 0.5, 0.25]).is_some());
    for diameter in [0, 1, 2, 4, 129, 131] {
        assert!(ImageMedian::new(&device, diameter).is_none(), "{diameter}");
    }
    for diameter in [3, 5, 127] {
        assert!(ImageMedian::new(&device, diameter).is_some(), "{diameter}");
    }
}

#[test]
fn histograms_check_their_source() {
    let device = device();
    let queue = queue(&device);
    let command_buffer = queue.new_command_buffer().expect("command buffer");
    let histogram = ImageHistogram::new(
        &device,
        HistogramInfo {
            number_of_entries: 256,
            histogram_for_alpha: false,
            min_pixel_value: [0.0; 4],
            max_pixel_value: [1.0; 4],
        },
    )
    .expect("histogram");
    for format in [pixel_format::A8UNORM, pixel_format::DEPTH32FLOAT, 9_999] {
        assert_eq!(
            histogram.histogram_size_for_source_format(format),
            Err(unsupported("source", format))
        );
    }
    assert_eq!(
        histogram.histogram_size_for_source_format(pixel_format::RGBA8UNORM),
        Ok(3 * 256 * 4)
    );
    let bins = device
        .new_buffer(4096, apple_metal::resource_options::STORAGE_MODE_SHARED)
        .expect("bins");
    let layered = array_texture(&device, pixel_format::RGBA8UNORM, 2);
    assert!(matches!(
        histogram.encode_texture(&command_buffer, &layered, &bins, 0),
        Err(Error::InvalidArgument(_))
    ));
    let wide = image(&device, 8);
    assert!(matches!(
        histogram.encode_image(&command_buffer, &wide, &bins, 0),
        Err(Error::InvalidArgument(_))
    ));
    let alpha = texture(&device, pixel_format::A8UNORM, 8, 8);
    assert_eq!(
        histogram.encode_texture(&command_buffer, &alpha, &bins, 0),
        Err(unsupported("source", pixel_format::A8UNORM))
    );
    let integer = texture(&device, pixel_format::R32UINT, 8, 8);
    histogram
        .encode_texture(&command_buffer, &integer, &bins, 0)
        .expect("histograms accept integer sources");
    run(&command_buffer);
}
