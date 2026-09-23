use apple_metal::{pixel_format, storage_mode, texture_usage, MetalDevice, TextureDescriptor};
use apple_mps::{
    feature_channel_format, image_layout, Error, Image, ImageDescriptor, ImageReadWriteParams,
    ImageRegion,
};

fn device() -> MetalDevice {
    MetalDevice::system_default().expect("no Metal device available")
}

fn float_image(device: &MetalDevice, width: usize, height: usize, channels: usize) -> Image {
    Image::new(
        device,
        ImageDescriptor::new(width, height, channels, feature_channel_format::FLOAT32),
    )
    .expect("image")
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn bytes_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

#[test]
fn row_stride_shorter_than_a_row_is_rejected() {
    let device = device();
    let image = float_image(&device, 4, 4, 1);
    let mut dst = [0_u8; 4];
    let error = image
        .read_bytes(
            &mut dst,
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            1,
            ImageRegion::whole(4, 4),
            ImageReadWriteParams::all(1),
            0,
        )
        .expect_err("a 1-byte row stride cannot hold 4 float pixels");
    assert_eq!(
        error,
        Error::DimensionMismatch {
            field: "bytes_per_row",
            expected: 16,
            actual: 1,
        }
    );
    let src = [0_u8; 4];
    let error = image
        .write_bytes(
            &src,
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            1,
            ImageRegion::whole(4, 4),
            ImageReadWriteParams::all(1),
            0,
        )
        .expect_err("write with a short row stride");
    assert!(matches!(
        error,
        Error::DimensionMismatch {
            field: "bytes_per_row",
            ..
        }
    ));
}

#[test]
fn planar_rows_need_one_row_per_channel_plane() {
    let device = device();
    let image = float_image(&device, 4, 2, 3);
    let mut dst = vec![0_u8; 4 * 2 * 3 * 4];
    let error = image
        .read_bytes(
            &mut dst,
            image_layout::FEATURE_CHANNELSxHEIGHTxWIDTH,
            8,
            ImageRegion::whole(4, 2),
            ImageReadWriteParams::all(3),
            0,
        )
        .expect_err("planar rows hold 4 floats");
    assert_eq!(
        error,
        Error::DimensionMismatch {
            field: "bytes_per_row",
            expected: 16,
            actual: 8,
        }
    );
}

#[test]
fn destination_shorter_than_the_layout_is_rejected() {
    let device = device();
    let image = float_image(&device, 4, 4, 1);
    let mut dst = vec![0_u8; 63];
    let error = image
        .read_bytes(
            &mut dst,
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            16,
            ImageRegion::whole(4, 4),
            ImageReadWriteParams::all(1),
            0,
        )
        .expect_err("64 bytes are needed");
    assert_eq!(
        error,
        Error::InvalidLength {
            expected: 64,
            actual: 63,
        }
    );
    let mut planar = vec![0_u8; 4 * 4 * 3 * 4 - 1];
    let image = float_image(&device, 4, 4, 3);
    let error = image
        .read_bytes(
            &mut planar,
            image_layout::FEATURE_CHANNELSxHEIGHTxWIDTH,
            16,
            ImageRegion::whole(4, 4),
            ImageReadWriteParams::all(3),
            0,
        )
        .expect_err("three planes of 64 bytes are needed");
    assert_eq!(
        error,
        Error::InvalidLength {
            expected: 192,
            actual: 191,
        }
    );
}

#[test]
fn channel_windows_follow_the_mps_rules() {
    let device = device();
    let image = float_image(&device, 1, 1, 6);
    let mut dst = [0_u8; 64];
    let read = |dst: &mut [u8], offset: usize, count: usize| {
        image.read_bytes(
            dst,
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            32,
            ImageRegion::whole(1, 1),
            ImageReadWriteParams::new(offset, count),
            0,
        )
    };
    assert_eq!(
        read(&mut dst, 2, 4),
        Err(Error::Misaligned {
            field: "feature_channel_offset",
            value: 2,
            alignment: 4,
        })
    );
    assert!(matches!(
        read(&mut dst, 0, 0),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        read(&mut dst, 0, 2),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        read(&mut dst, 4, 4),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        read(&mut dst, 0, usize::MAX),
        Err(Error::InvalidArgument(_))
    ));

    let values = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    image
        .write_bytes(
            &f32_bytes(&values),
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            24,
            ImageRegion::whole(1, 1),
            ImageReadWriteParams::all(6),
            0,
        )
        .expect("write all channels");
    let mut tail = [0_u8; 8];
    image
        .read_bytes(
            &mut tail,
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            8,
            ImageRegion::whole(1, 1),
            ImageReadWriteParams::new(4, 2),
            0,
        )
        .expect("the trailing window may be shorter than 4");
    assert_eq!(bytes_f32(&tail), vec![5.0, 6.0]);
}

#[test]
fn regions_and_image_indices_must_fit() {
    let device = device();
    let image = Image::new(
        &device,
        ImageDescriptor {
            number_of_images: 2,
            ..ImageDescriptor::new(2, 2, 1, feature_channel_format::FLOAT32)
        },
    )
    .expect("image batch");
    let mut dst = vec![0_u8; 256];
    let mut read = |region: ImageRegion, layout: usize, image_index: usize| {
        image.read_bytes(
            &mut dst,
            layout,
            8,
            region,
            ImageReadWriteParams::all(1),
            image_index,
        )
    };
    let chunky = image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS;
    let expect_invalid = |result: apple_mps::Result<()>| {
        assert!(
            matches!(result, Err(Error::InvalidArgument(_))),
            "{result:?}"
        );
    };
    expect_invalid(read(ImageRegion::new(1, 0, 0, 2, 2, 1), chunky, 0));
    expect_invalid(read(ImageRegion::new(0, 1, 0, 2, 2, 1), chunky, 0));
    expect_invalid(read(ImageRegion::new(usize::MAX, 0, 0, 2, 2, 1), chunky, 0));
    expect_invalid(read(ImageRegion::new(0, 0, 1, 2, 2, 1), chunky, 0));
    expect_invalid(read(ImageRegion::new(0, 0, 0, 2, 2, 2), chunky, 0));
    expect_invalid(read(ImageRegion::new(0, 0, 0, 0, 2, 1), chunky, 0));
    expect_invalid(read(ImageRegion::whole(2, 2), chunky, 2));
    expect_invalid(read(ImageRegion::whole(2, 2), 7, 0));
    read(ImageRegion::whole(2, 2), chunky, 1).expect("second image");
}

#[test]
fn chunky_and_planar_layouts_round_trip() {
    let device = device();
    let image = float_image(&device, 2, 2, 3);
    let chunky: Vec<f32> = (0_u8..12).map(f32::from).collect();
    image
        .write_bytes(
            &f32_bytes(&chunky),
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            24,
            ImageRegion::whole(2, 2),
            ImageReadWriteParams::all(3),
            0,
        )
        .expect("write chunky");
    let mut planar = vec![0_u8; 48];
    image
        .read_bytes(
            &mut planar,
            image_layout::FEATURE_CHANNELSxHEIGHTxWIDTH,
            8,
            ImageRegion::whole(2, 2),
            ImageReadWriteParams::all(3),
            0,
        )
        .expect("read planar");
    assert_eq!(
        bytes_f32(&planar),
        vec![0.0, 3.0, 6.0, 9.0, 1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0]
    );
    assert_eq!(image.read_f32().expect("read_f32"), chunky);
}

#[test]
fn f32_helpers_need_a_float32_image() {
    let device = device();
    let image = Image::new(
        &device,
        ImageDescriptor::new(2, 2, 1, feature_channel_format::UNORM8),
    )
    .expect("unorm8 image");
    assert_eq!(
        image.feature_channel_format(),
        feature_channel_format::UNORM8
    );
    assert!(matches!(image.read_f32(), Err(Error::Unsupported(_))));
    assert!(matches!(
        image.write_f32(&[0.0; 4]),
        Err(Error::Unsupported(_))
    ));
    let mut dst = [0_u8; 4];
    image
        .read_bytes(
            &mut dst,
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            2,
            ImageRegion::whole(2, 2),
            ImageReadWriteParams::all(1),
            0,
        )
        .expect("unorm8 rows are 2 bytes");
}

#[test]
fn unwritten_images_can_be_read() {
    let device = device();
    let image = float_image(&device, 3, 2, 2);
    let values = image.read_f32().expect("reading an unwritten image");
    assert_eq!(values.len(), 12);
}

#[test]
fn private_images_refuse_cpu_transfers() {
    let device = device();
    let image = Image::new(
        &device,
        ImageDescriptor {
            storage_mode: storage_mode::PRIVATE,
            ..ImageDescriptor::new(2, 2, 1, feature_channel_format::FLOAT32)
        },
    )
    .expect("private image");
    assert_eq!(
        image.read_f32(),
        Err(Error::Rejected("MPSImage byte transfer"))
    );
}

#[test]
fn invalid_image_descriptors_are_refused() {
    let device = device();
    let valid = ImageDescriptor::new(2, 2, 1, feature_channel_format::FLOAT32);
    let invalid = [
        ImageDescriptor { width: 0, ..valid },
        ImageDescriptor { height: 0, ..valid },
        ImageDescriptor {
            width: 16_385,
            ..valid
        },
        ImageDescriptor {
            feature_channels: 0,
            ..valid
        },
        ImageDescriptor {
            number_of_images: 0,
            ..valid
        },
        ImageDescriptor {
            feature_channels: 8,
            number_of_images: 1_025,
            ..valid
        },
        ImageDescriptor {
            channel_format: feature_channel_format::NONE,
            ..valid
        },
        ImageDescriptor {
            channel_format: 9,
            ..valid
        },
        ImageDescriptor {
            storage_mode: storage_mode::MEMORYLESS,
            ..valid
        },
        ImageDescriptor {
            usage: 0x40,
            ..valid
        },
    ];
    for descriptor in invalid {
        assert!(
            Image::new(&device, descriptor).is_none(),
            "{descriptor:?} should be refused"
        );
    }
    let image = Image::new(&device, valid).expect("valid descriptor");
    assert_eq!(image.width(), 2);
    assert_eq!(image.feature_channels(), 1);
}

#[test]
fn texture_backed_images_must_match_their_texture() {
    let device = device();
    let texture_with = |format: usize| {
        device
            .new_texture(TextureDescriptor {
                usage: texture_usage::SHADER_READ | texture_usage::SHADER_WRITE,
                ..TextureDescriptor::new_2d(2, 2, format)
            })
            .expect("texture")
    };
    let rgba = texture_with(pixel_format::RGBA32FLOAT);
    assert!(Image::from_texture(&rgba, 0).is_none());
    assert!(Image::from_texture(&rgba, 1).is_none());
    assert!(Image::from_texture(&rgba, 9).is_none());
    let image = Image::from_texture(&rgba, 4).expect("four channels");
    assert_eq!(image.feature_channels(), 4);
    assert_eq!(
        image.feature_channel_format(),
        feature_channel_format::FLOAT32
    );
    let red = texture_with(pixel_format::R32FLOAT);
    assert!(Image::from_texture(&red, 2).is_none());
    assert_eq!(
        Image::from_texture(&red, 1)
            .expect("one channel")
            .feature_channels(),
        1
    );
    let integer = texture_with(pixel_format::R8UINT);
    assert!(Image::from_texture(&integer, 1).is_none());
}

#[test]
fn the_bridge_refuses_layouts_rust_would_reject() {
    let device = device();
    let image = float_image(&device, 4, 4, 1);
    let mut dst = [0_u8; 4];
    let accepted = unsafe {
        apple_mps::ffi::mps_image_read_bytes(
            image.as_ptr(),
            dst.as_mut_ptr().cast(),
            dst.len(),
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            1,
            0,
            0,
            0,
            4,
            4,
            1,
            0,
            1,
            0,
        )
    };
    assert!(!accepted);
    let mut full = [0_u8; 64];
    let accepted = unsafe {
        apple_mps::ffi::mps_image_read_bytes(
            image.as_ptr(),
            full.as_mut_ptr().cast(),
            full.len(),
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            16,
            0,
            0,
            0,
            4,
            4,
            1,
            0,
            1,
            0,
        )
    };
    assert!(accepted);
}
