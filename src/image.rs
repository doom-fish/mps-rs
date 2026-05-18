use crate::error::{Error, Result};
use crate::ffi;
use apple_metal::{storage_mode, texture_usage, MetalDevice, MetalTexture};
use core::ffi::c_void;
use core::ptr;

/// `MPSImageFeatureChannelFormat` constants.
pub mod feature_channel_format {
    /// Wraps a `MPSImageFeatureChannelFormat` raw value.
    pub const NONE: usize = 0;
    /// Wraps a `MPSImageFeatureChannelFormat` raw value.
    pub const UNORM8: usize = 1;
    /// Wraps a `MPSImageFeatureChannelFormat` raw value.
    pub const UNORM16: usize = 2;
    /// Wraps a `MPSImageFeatureChannelFormat` raw value.
    pub const FLOAT16: usize = 3;
    /// Wraps a `MPSImageFeatureChannelFormat` raw value.
    pub const FLOAT32: usize = 4;
}

/// `MPSDataLayout` constants.
#[allow(non_upper_case_globals)]
pub mod image_layout {
    /// Wraps a `MPSDataLayout` raw value.
    pub const HEIGHTxWIDTHxFEATURE_CHANNELS: usize = 0;
    /// Wraps a `MPSDataLayout` raw value.
    pub const FEATURE_CHANNELSxHEIGHTxWIDTH: usize = 1;
}

/// `MPSImageEdgeMode` constants.
pub mod image_edge_mode {
    /// Wraps a `MPSImageEdgeMode` raw value.
    pub const ZERO: usize = 0;
    /// Wraps a `MPSImageEdgeMode` raw value.
    pub const CLAMP: usize = 1;
}

/// `MPSKernelOptions` constants.
pub mod kernel_options {
    /// Wraps a `MPSKernelOptions` raw value.
    pub const NONE: u32 = 0;
    /// Wraps a `MPSKernelOptions` raw value.
    pub const SKIP_API_VALIDATION: u32 = 1 << 0;
    /// Wraps a `MPSKernelOptions` raw value.
    pub const ALLOW_REDUCED_PRECISION: u32 = 1 << 1;
    /// Wraps a `MPSKernelOptions` raw value.
    pub const DISABLE_INTERNAL_TILING: u32 = 1 << 2;
    /// Wraps a `MPSKernelOptions` raw value.
    pub const INSERT_DEBUG_GROUPS: u32 = 1 << 3;
    /// Wraps a `MPSKernelOptions` raw value.
    pub const VERBOSE: u32 = 1 << 4;
}

/// Plain-Rust configuration for building a `MPSImageDescriptor` on the Swift side.
#[derive(Debug, Clone, Copy)]
pub struct ImageDescriptor {
    /// Corresponds to the `channel_format` field on `MPSImageDescriptor`.
    pub channel_format: usize,
    /// Corresponds to the `width` field on `MPSImageDescriptor`.
    pub width: usize,
    /// Corresponds to the `height` field on `MPSImageDescriptor`.
    pub height: usize,
    /// Corresponds to the `feature_channels` field on `MPSImageDescriptor`.
    pub feature_channels: usize,
    /// Corresponds to the `number_of_images` field on `MPSImageDescriptor`.
    pub number_of_images: usize,
    /// Corresponds to the `usage` field on `MPSImageDescriptor`.
    pub usage: usize,
    /// Corresponds to the `storage_mode` field on `MPSImageDescriptor`.
    pub storage_mode: usize,
}

impl ImageDescriptor {
    /// Create a single-image descriptor with sensible defaults for read/write image processing.
    #[must_use]
    pub const fn new(
        width: usize,
        height: usize,
        feature_channels: usize,
        channel_format: usize,
    ) -> Self {
        Self {
            channel_format,
            width,
            height,
            feature_channels,
            number_of_images: 1,
            usage: texture_usage::SHADER_READ | texture_usage::SHADER_WRITE,
            storage_mode: storage_mode::MANAGED,
        }
    }
}

/// Rectangular region used for image transfer or clip-rect configuration.
#[derive(Debug, Clone, Copy)]
pub struct ImageRegion {
    /// Corresponds to the `x` field on `MPSRegion`.
    pub x: usize,
    /// Corresponds to the `y` field on `MPSRegion`.
    pub y: usize,
    /// Corresponds to the `z` field on `MPSRegion`.
    pub z: usize,
    /// Corresponds to the `width` field on `MPSRegion`.
    pub width: usize,
    /// Corresponds to the `height` field on `MPSRegion`.
    pub height: usize,
    /// Corresponds to the `depth` field on `MPSRegion`.
    pub depth: usize,
}

impl ImageRegion {
    /// Construct an arbitrary region.
    #[must_use]
    pub const fn new(
        x: usize,
        y: usize,
        z: usize,
        width: usize,
        height: usize,
        depth: usize,
    ) -> Self {
        Self {
            x,
            y,
            z,
            width,
            height,
            depth,
        }
    }

    /// Region covering the full first image slice.
    #[must_use]
    pub const fn whole(width: usize, height: usize) -> Self {
        Self::new(0, 0, 0, width, height, 1)
    }
}

/// `MPSImageReadWriteParams` values.
#[derive(Debug, Clone, Copy)]
pub struct ImageReadWriteParams {
    /// Corresponds to the `feature_channel_offset` field on `MPSImageReadWriteParams`.
    pub feature_channel_offset: usize,
    /// Corresponds to the `feature_channel_count` field on `MPSImageReadWriteParams`.
    pub feature_channel_count: usize,
}

impl ImageReadWriteParams {
    /// Create a parameter block describing the feature-channel window to transfer.
    #[must_use]
    pub const fn new(feature_channel_offset: usize, feature_channel_count: usize) -> Self {
        Self {
            feature_channel_offset,
            feature_channel_count,
        }
    }

    /// Transfer all feature channels starting at zero.
    #[must_use]
    pub const fn all(feature_channels: usize) -> Self {
        Self::new(0, feature_channels)
    }
}

/// Safe owner for an Objective-C `MPSImage`.
pub struct Image {
    ptr: *mut c_void,
}

// SAFETY: MPSImage pointers are thread-safe Objective-C objects.
unsafe impl Send for Image {}
// SAFETY: MPSImage pointers are thread-safe Objective-C objects.
unsafe impl Sync for Image {}

impl Drop for Image {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
            unsafe { ffi::mps_object_release(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl Image {
    /// Allocate a lazily backed `MPSImage` on `device`.
    #[must_use]
    pub fn new(device: &MetalDevice, descriptor: ImageDescriptor) -> Option<Self> {
        // SAFETY: All pointers originate from safe wrappers and the scalar arguments are POD.
        let ptr = unsafe {
            ffi::mps_image_new_with_descriptor(
                device.as_ptr(),
                descriptor.channel_format,
                descriptor.width,
                descriptor.height,
                descriptor.feature_channels,
                descriptor.number_of_images,
                descriptor.usage,
                descriptor.storage_mode,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wrap an existing Metal texture in an `MPSImage`.
    #[must_use]
    pub fn from_texture(texture: &MetalTexture, feature_channels: usize) -> Option<Self> {
        // SAFETY: `texture` is a valid `MTLTexture` pointer from `apple-metal`.
        let ptr = unsafe { ffi::mps_image_new_with_texture(texture.as_ptr(), feature_channels) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Raw `MPSImage` pointer.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    #[must_use]
    pub(crate) const unsafe fn from_raw(ptr: *mut c_void) -> Self {
        // SAFETY: Caller must ensure `ptr` is a valid +1 retained MPSImage pointer.
        // SAFETY: Caller must ensure `ptr` is a valid +1 retained MPSImage pointer.
        Self { ptr }
    }

    /// Image width in pixels.
    #[must_use]
    pub fn width(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSImage` pointer while `self` is alive.
        unsafe { ffi::mps_image_width(self.ptr) }
    }

    /// Image height in pixels.
    #[must_use]
    pub fn height(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSImage` pointer while `self` is alive.
        unsafe { ffi::mps_image_height(self.ptr) }
    }

    /// Number of feature channels per pixel.
    #[must_use]
    pub fn feature_channels(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSImage` pointer while `self` is alive.
        unsafe { ffi::mps_image_feature_channels(self.ptr) }
    }

    /// Number of images stored in the backing texture array.
    #[must_use]
    pub fn number_of_images(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSImage` pointer while `self` is alive.
        unsafe { ffi::mps_image_number_of_images(self.ptr) }
    }

    /// Bytes between neighboring pixels in storage order.
    #[must_use]
    pub fn pixel_size(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSImage` pointer while `self` is alive.
        unsafe { ffi::mps_image_pixel_size(self.ptr) }
    }

    /// Underlying `MTLPixelFormat` raw value.
    #[must_use]
    pub fn pixel_format(&self) -> usize {
        // SAFETY: `self.ptr` is a valid `MPSImage` pointer while `self` is alive.
        unsafe { ffi::mps_image_pixel_format(self.ptr) }
    }

    /// Convenience region covering the full first image.
    #[must_use]
    pub fn whole_region(&self) -> ImageRegion {
        ImageRegion::whole(self.width(), self.height())
    }

    /// Read bytes out of the image into a caller-provided buffer.
    pub fn read_bytes(
        &self,
        dst: &mut [u8],
        data_layout: usize,
        bytes_per_row: usize,
        region: ImageRegion,
        params: ImageReadWriteParams,
        image_index: usize,
    ) -> Result<()> {
        let expected = required_bytes(data_layout, bytes_per_row, region, params);
        if dst.len() < expected {
            return Err(Error::InvalidLength {
                expected,
                actual: dst.len(),
            });
        }

        // SAFETY: `dst` is valid for writes of at least `expected` bytes and all handles are valid.
        let _ = unsafe {
            ffi::mps_image_read_bytes(
                self.ptr,
                dst.as_mut_ptr().cast(),
                data_layout,
                bytes_per_row,
                region.x,
                region.y,
                region.z,
                region.width,
                region.height,
                region.depth,
                params.feature_channel_offset,
                params.feature_channel_count,
                image_index,
            )
        };
        Ok(())
    }

    /// Write bytes into the image from a caller-provided buffer.
    pub fn write_bytes(
        &self,
        src: &[u8],
        data_layout: usize,
        bytes_per_row: usize,
        region: ImageRegion,
        params: ImageReadWriteParams,
        image_index: usize,
    ) -> Result<()> {
        let expected = required_bytes(data_layout, bytes_per_row, region, params);
        if src.len() < expected {
            return Err(Error::InvalidLength {
                expected,
                actual: src.len(),
            });
        }

        // SAFETY: `src` is valid for reads of at least `expected` bytes and all handles are valid.
        let _ = unsafe {
            ffi::mps_image_write_bytes(
                self.ptr,
                src.as_ptr().cast(),
                data_layout,
                bytes_per_row,
                region.x,
                region.y,
                region.z,
                region.width,
                region.height,
                region.depth,
                params.feature_channel_offset,
                params.feature_channel_count,
                image_index,
            )
        };
        Ok(())
    }

    /// Read the first image slice as tightly packed float32 HWC data.
    pub fn read_f32(&self) -> Result<Vec<f32>> {
        let len = self.width() * self.height() * self.feature_channels();
        let mut data = vec![0.0_f32; len];
        let bytes_per_row = self.width() * self.feature_channels() * core::mem::size_of::<f32>();
        // SAFETY: `data` is a contiguous `Vec<f32>` with exactly `len * size_of::<f32>()` bytes.
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(
                data.as_mut_ptr().cast::<u8>(),
                core::mem::size_of_val(data.as_slice()),
            )
        };
        self.read_bytes(
            bytes,
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            bytes_per_row,
            self.whole_region(),
            ImageReadWriteParams::all(self.feature_channels()),
            0,
        )?;
        Ok(data)
    }

    /// Write tightly packed float32 HWC data into the first image slice.
    pub fn write_f32(&self, data: &[f32]) -> Result<()> {
        let expected = self.width() * self.height() * self.feature_channels();
        if data.len() != expected {
            return Err(Error::InvalidLength {
                expected: expected * core::mem::size_of::<f32>(),
                actual: core::mem::size_of_val(data),
            });
        }

        let bytes_per_row = self.width() * self.feature_channels() * core::mem::size_of::<f32>();
        // SAFETY: `data` is a contiguous slice of `f32`, which may be viewed as bytes.
        let bytes = unsafe {
            core::slice::from_raw_parts(data.as_ptr().cast::<u8>(), core::mem::size_of_val(data))
        };
        self.write_bytes(
            bytes,
            image_layout::HEIGHTxWIDTHxFEATURE_CHANNELS,
            bytes_per_row,
            self.whole_region(),
            ImageReadWriteParams::all(self.feature_channels()),
            0,
        )
    }
}

#[doc(hidden)]
pub use crate::generated::image::*;

fn required_bytes(
    data_layout: usize,
    bytes_per_row: usize,
    region: ImageRegion,
    params: ImageReadWriteParams,
) -> usize {
    let rows = region.height.saturating_mul(region.depth);
    let base = bytes_per_row.saturating_mul(rows);
    if data_layout == image_layout::FEATURE_CHANNELSxHEIGHTxWIDTH {
        base.saturating_mul(params.feature_channel_count.max(1))
    } else {
        base
    }
}
