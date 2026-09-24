use crate::error::{Error, Result};
use crate::ffi;
use crate::image::{Image, ImageRegion};
use apple_metal::{pixel_format as pf, texture_type, texture_usage, MetalTexture};
use core::ffi::c_void;

#[derive(Clone, Copy, PartialEq, Eq)]
enum ColorModel {
    Monochrome,
    Alpha,
    TwoChannel,
    Rgb,
}

#[derive(Clone, Copy)]
struct FormatTraits {
    channels: u8,
    model: ColorModel,
    integer: bool,
    writable: bool,
}

const fn format_traits(pixel_format: usize) -> Option<FormatTraits> {
    use ColorModel::{Alpha, Monochrome, Rgb, TwoChannel};
    let (channels, model, integer, writable) = match pixel_format {
        pf::A8UNORM => (1, Alpha, false, true),
        pf::R8UNORM
        | pf::R8UNORM_SRGB
        | pf::R8SNORM
        | pf::R16UNORM
        | pf::R16SNORM
        | pf::R16FLOAT
        | pf::R32FLOAT => (1, Monochrome, false, true),
        pf::R8UINT | pf::R8SINT | pf::R16UINT | pf::R16SINT | pf::R32UINT | pf::R32SINT => {
            (1, Monochrome, true, true)
        }
        pf::RG8UNORM
        | pf::RG8UNORM_SRGB
        | pf::RG8SNORM
        | pf::RG16UNORM
        | pf::RG16SNORM
        | pf::RG16FLOAT
        | pf::RG32FLOAT => (2, TwoChannel, false, true),
        pf::RG8UINT | pf::RG8SINT | pf::RG16UINT | pf::RG16SINT | pf::RG32UINT | pf::RG32SINT => {
            (2, TwoChannel, true, true)
        }
        pf::RG11B10FLOAT | pf::RGB9E5FLOAT | pf::BGR10_XR | pf::BGR10_XR_SRGB => {
            (3, Rgb, false, true)
        }
        pf::B5G6R5UNORM => (3, Rgb, false, false),
        pf::RGBA8UNORM
        | pf::RGBA8UNORM_SRGB
        | pf::RGBA8SNORM
        | pf::BGRA8UNORM
        | pf::BGRA8UNORM_SRGB
        | pf::RGBA16UNORM
        | pf::RGBA16SNORM
        | pf::RGBA16FLOAT
        | pf::RGBA32FLOAT
        | pf::RGB10A2UNORM
        | pf::BGR10A2UNORM
        | pf::BGRA10_XR
        | pf::BGRA10_XR_SRGB => (4, Rgb, false, true),
        pf::A1BGR5UNORM | pf::ABGR4UNORM | pf::BGR5A1UNORM => (4, Rgb, false, false),
        pf::RGBA8UINT
        | pf::RGBA8SINT
        | pf::RGBA16UINT
        | pf::RGBA16SINT
        | pf::RGBA32UINT
        | pf::RGBA32SINT
        | pf::RGB10A2UINT => (4, Rgb, true, true),
        _ => return None,
    };
    Some(FormatTraits {
        channels,
        model,
        integer,
        writable,
    })
}

pub fn histogram_source_supported(pixel_format: usize) -> bool {
    format_traits(pixel_format).is_some_and(|traits| traits.model != ColorModel::Alpha)
}

#[derive(Clone, Copy)]
pub struct Operand {
    texture: *mut c_void,
    pixel_format: usize,
    width: usize,
    height: usize,
    texture_type: usize,
    array_length: usize,
    usage: usize,
    feature_channels: Option<usize>,
}

impl Operand {
    pub fn texture(texture: &MetalTexture) -> Self {
        Self {
            texture: texture.as_ptr(),
            pixel_format: texture.pixel_format(),
            width: texture.width(),
            height: texture.height(),
            texture_type: texture.texture_type(),
            array_length: texture.array_length(),
            usage: texture.usage(),
            feature_channels: None,
        }
    }

    pub fn image(image: &Image) -> Result<Self> {
        let mut texture = core::ptr::null_mut();
        let mut texture_type = 0;
        let mut array_length = 0;
        let mut usage = 0;
        // SAFETY: `image` is a live MPSImage and the out-pointers are valid for the call.
        let found = unsafe {
            ffi::mps_image_texture_info(
                image.as_ptr(),
                &raw mut texture,
                &raw mut texture_type,
                &raw mut array_length,
                &raw mut usage,
            )
        };
        if !found || texture.is_null() {
            return Err(Error::Rejected("MPSImage texture query"));
        }
        Ok(Self {
            texture,
            pixel_format: image.pixel_format(),
            width: image.width(),
            height: image.height(),
            texture_type,
            array_length,
            usage,
            feature_channels: Some(image.feature_channels()),
        })
    }
}

#[derive(Clone, Copy)]
pub enum ChannelRule {
    SameColorModel,
    SameChannelsOrMonochromeDestination,
    Any,
    SameChannelCount,
}

#[derive(Clone, Copy)]
pub struct UnaryRules {
    channels: ChannelRule,
    alpha: bool,
    min_clip_width: usize,
}

pub const SPATIAL: UnaryRules = UnaryRules {
    channels: ChannelRule::SameColorModel,
    alpha: true,
    min_clip_width: 0,
};
pub const MEDIAN: UnaryRules = UnaryRules {
    alpha: false,
    ..SPATIAL
};
pub const SOBEL: UnaryRules = UnaryRules {
    channels: ChannelRule::SameChannelsOrMonochromeDestination,
    ..SPATIAL
};
pub const THRESHOLD: UnaryRules = UnaryRules {
    channels: ChannelRule::Any,
    ..SPATIAL
};
pub const MEAN: UnaryRules = UnaryRules {
    channels: ChannelRule::SameChannelCount,
    alpha: false,
    min_clip_width: 0,
};
pub const MIN_AND_MAX: UnaryRules = UnaryRules {
    min_clip_width: 2,
    ..MEAN
};

const MAX_UNARY_FEATURE_CHANNELS: usize = 4;

impl UnaryRules {
    pub fn needs_clip_rect(self) -> bool {
        self.min_clip_width > 0
    }

    pub fn check(
        self,
        source: &Operand,
        destination: &Operand,
        clip_rect: Option<ImageRegion>,
    ) -> Result<()> {
        let source_traits = filterable(source, "source", self.alpha)?;
        let destination_traits = filterable(destination, "destination", self.alpha)?;
        ensure_writable(destination, destination_traits)?;
        ensure_readable(source)?;
        ensure_distinct(source, destination)?;
        ensure_same_type(source, destination)?;
        if source.texture_type == texture_type::TYPE_2D_ARRAY
            && source.array_length != destination.array_length
        {
            return Err(Error::DimensionMismatch {
                field: "destination array_length",
                expected: source.array_length,
                actual: destination.array_length,
            });
        }
        for operand in [source, destination] {
            if operand
                .feature_channels
                .is_some_and(|channels| channels > MAX_UNARY_FEATURE_CHANNELS)
            {
                return Err(Error::InvalidArgument(
                    "unary image kernels take images with at most 4 feature channels",
                ));
            }
        }
        let compatible = match self.channels {
            ChannelRule::SameColorModel => source_traits.model == destination_traits.model,
            ChannelRule::SameChannelsOrMonochromeDestination => {
                destination_traits.model == ColorModel::Monochrome
                    || (source_traits.model == destination_traits.model
                        && source_traits.channels == destination_traits.channels)
            }
            ChannelRule::Any => true,
            ChannelRule::SameChannelCount => source_traits.channels == destination_traits.channels,
        };
        if !compatible {
            return Err(Error::InvalidArgument(match self.channels {
                ChannelRule::SameChannelsOrMonochromeDestination => {
                    "the destination needs the source's channels or a single channel"
                }
                ChannelRule::SameChannelCount => {
                    "the destination needs as many channels as the source"
                }
                _ => "source and destination color models differ; the kernel does not convert them",
            }));
        }
        if self.needs_clip_rect() {
            let clip =
                clip_rect.ok_or(Error::Rejected("MPSUnaryImageKernel clip rectangle query"))?;
            let width = clipped_extent(clip.x, clip.width, destination.width);
            let height = clipped_extent(clip.y, clip.height, destination.height);
            if width < self.min_clip_width || height == 0 {
                return Err(Error::DimensionMismatch {
                    field: "clipped destination width",
                    expected: self.min_clip_width,
                    actual: width,
                });
            }
        }
        Ok(())
    }
}

pub fn check_binary(primary: &Operand, secondary: &Operand, destination: &Operand) -> Result<()> {
    let primary_traits = filterable(primary, "primary", true)?;
    filterable(secondary, "secondary", true)?;
    let destination_traits = filterable(destination, "destination", true)?;
    ensure_writable(destination, destination_traits)?;
    ensure_readable(primary)?;
    ensure_readable(secondary)?;
    ensure_distinct(primary, destination)?;
    ensure_distinct(secondary, destination)?;
    ensure_same_type(primary, secondary)?;
    ensure_same_type(primary, destination)?;
    if primary_traits.model != destination_traits.model {
        return Err(Error::InvalidArgument(
            "primary and destination color models differ; the kernel does not convert them",
        ));
    }
    Ok(())
}

pub fn check_histogram_source(source: &Operand) -> Result<()> {
    if !histogram_source_supported(source.pixel_format) {
        return Err(Error::UnsupportedPixelFormat {
            operand: "source",
            pixel_format: source.pixel_format,
        });
    }
    if source.texture_type != texture_type::TYPE_2D {
        return Err(Error::InvalidArgument(
            "MPSImageHistogram needs a 2D source texture",
        ));
    }
    ensure_readable(source)
}

fn filterable(operand: &Operand, name: &'static str, alpha: bool) -> Result<FormatTraits> {
    format_traits(operand.pixel_format)
        .filter(|traits| !traits.integer && (alpha || traits.model != ColorModel::Alpha))
        .ok_or(Error::UnsupportedPixelFormat {
            operand: name,
            pixel_format: operand.pixel_format,
        })
}

fn ensure_writable(destination: &Operand, traits: FormatTraits) -> Result<()> {
    if !traits.writable {
        return Err(Error::UnsupportedPixelFormat {
            operand: "destination",
            pixel_format: destination.pixel_format,
        });
    }
    if destination.usage & texture_usage::SHADER_WRITE == 0 {
        return Err(Error::InvalidArgument(
            "the destination texture needs texture_usage::SHADER_WRITE",
        ));
    }
    Ok(())
}

fn ensure_readable(source: &Operand) -> Result<()> {
    if source.usage & texture_usage::SHADER_READ == 0 {
        return Err(Error::InvalidArgument(
            "source textures need texture_usage::SHADER_READ",
        ));
    }
    Ok(())
}

fn ensure_distinct(source: &Operand, destination: &Operand) -> Result<()> {
    if source.texture == destination.texture {
        return Err(Error::InvalidArgument(
            "the destination may not alias a source; these kernels do not run in place",
        ));
    }
    Ok(())
}

fn ensure_same_type(first: &Operand, second: &Operand) -> Result<()> {
    if !matches!(
        first.texture_type,
        texture_type::TYPE_2D | texture_type::TYPE_2D_ARRAY
    ) || first.texture_type != second.texture_type
    {
        return Err(Error::InvalidArgument(
            "image kernels need 2D or 2D-array textures of the same type",
        ));
    }
    Ok(())
}

fn clipped_extent(origin: usize, size: usize, limit: usize) -> usize {
    origin
        .saturating_add(size)
        .min(limit)
        .saturating_sub(origin)
}
