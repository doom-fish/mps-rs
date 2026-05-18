use crate::ffi;
use crate::image::{Image, ImageRegion};
use apple_metal::{CommandBuffer, MetalBuffer, MetalDevice, MetalTexture};
use core::ffi::c_void;
use core::ptr;

macro_rules! opaque_handle {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        pub struct $name {
            ptr: *mut c_void,
        }

        // SAFETY: MPS filter handles are opaque pointers to thread-safe Swift/ObjC objects.
        unsafe impl Send for $name {}
        // SAFETY: MPS filter handles are opaque pointers to thread-safe Swift/ObjC objects.
        unsafe impl Sync for $name {}

        impl Drop for $name {
            fn drop(&mut self) {
                if !self.ptr.is_null() {
                    // SAFETY: `ptr` is a +1 retained Swift/ObjC object pointer owned by this wrapper.
                    unsafe { ffi::mps_object_release(self.ptr) };
                    self.ptr = ptr::null_mut();
                }
            }
        }

        impl $name {
            /// Returns the retained Objective-C pointer backing this wrapper.
            #[must_use]
            pub const fn as_ptr(&self) -> *mut c_void {
                self.ptr
            }
        }
    };
}

macro_rules! impl_unary_methods {
    ($name:ident) => {
        impl $name {
            /// Encode the filter against `MPSImage` inputs/outputs.
            pub fn encode_image(
                &self,
                command_buffer: &CommandBuffer,
                source: &Image,
                destination: &Image,
            ) {
                // SAFETY: All handles come from safe wrappers and remain alive for the call.
                unsafe {
                    ffi::mps_unary_encode_image(
                        self.ptr,
                        command_buffer.as_ptr(),
                        source.as_ptr(),
                        destination.as_ptr(),
                    )
                };
            }

            /// Encode the filter directly against `MTLTexture` inputs/outputs.
            pub fn encode_texture(
                &self,
                command_buffer: &CommandBuffer,
                source: &MetalTexture,
                destination: &MetalTexture,
            ) {
                // SAFETY: All handles come from safe wrappers and remain alive for the call.
                unsafe {
                    ffi::mps_unary_encode_texture(
                        self.ptr,
                        command_buffer.as_ptr(),
                        source.as_ptr(),
                        destination.as_ptr(),
                    )
                };
            }

            /// Configure the kernel's edge mode.
            pub fn set_edge_mode(&self, edge_mode: usize) {
                // SAFETY: The kernel pointer is valid for the duration of the call.
                unsafe { ffi::mps_unary_set_edge_mode(self.ptr, edge_mode) };
            }

            /// Restrict writes to a destination clip rectangle.
            pub fn set_clip_rect(&self, region: ImageRegion) {
                // SAFETY: The kernel pointer is valid for the duration of the call.
                unsafe {
                    ffi::mps_unary_set_clip_rect(
                        self.ptr,
                        region.x,
                        region.y,
                        region.z,
                        region.width,
                        region.height,
                        region.depth,
                    )
                };
            }
        }
    };
}

macro_rules! impl_binary_methods {
    ($name:ident) => {
        impl $name {
            /// Encode the filter against `MPSImage` inputs/outputs.
            pub fn encode_image(
                &self,
                command_buffer: &CommandBuffer,
                primary: &Image,
                secondary: &Image,
                destination: &Image,
            ) {
                // SAFETY: All handles come from safe wrappers and remain alive for the call.
                unsafe {
                    ffi::mps_binary_encode_image(
                        self.ptr,
                        command_buffer.as_ptr(),
                        primary.as_ptr(),
                        secondary.as_ptr(),
                        destination.as_ptr(),
                    )
                };
            }

            /// Encode the filter directly against `MTLTexture` inputs/outputs.
            pub fn encode_texture(
                &self,
                command_buffer: &CommandBuffer,
                primary: &MetalTexture,
                secondary: &MetalTexture,
                destination: &MetalTexture,
            ) {
                // SAFETY: All handles come from safe wrappers and remain alive for the call.
                unsafe {
                    ffi::mps_binary_encode_texture(
                        self.ptr,
                        command_buffer.as_ptr(),
                        primary.as_ptr(),
                        secondary.as_ptr(),
                        destination.as_ptr(),
                    )
                };
            }

            /// Configure the primary input edge mode.
            pub fn set_primary_edge_mode(&self, edge_mode: usize) {
                // SAFETY: The kernel pointer is valid for the duration of the call.
                unsafe { ffi::mps_binary_set_primary_edge_mode(self.ptr, edge_mode) };
            }

            /// Configure the secondary input edge mode.
            pub fn set_secondary_edge_mode(&self, edge_mode: usize) {
                // SAFETY: The kernel pointer is valid for the duration of the call.
                unsafe { ffi::mps_binary_set_secondary_edge_mode(self.ptr, edge_mode) };
            }

            /// Restrict writes to a destination clip rectangle.
            pub fn set_clip_rect(&self, region: ImageRegion) {
                // SAFETY: The kernel pointer is valid for the duration of the call.
                unsafe {
                    ffi::mps_binary_set_clip_rect(
                        self.ptr,
                        region.x,
                        region.y,
                        region.z,
                        region.width,
                        region.height,
                        region.depth,
                    )
                };
            }
        }
    };
}

/// `MPSScaleTransform` values used by resampling kernels.
#[derive(Debug, Clone, Copy)]
pub struct ScaleTransform {
    /// Corresponds to the `scale_x` field on `MPSScaleTransform`.
    pub scale_x: f64,
    /// Corresponds to the `scale_y` field on `MPSScaleTransform`.
    pub scale_y: f64,
    /// Corresponds to the `translate_x` field on `MPSScaleTransform`.
    pub translate_x: f64,
    /// Corresponds to the `translate_y` field on `MPSScaleTransform`.
    pub translate_y: f64,
}

/// Plain-Rust configuration for `MPSImageHistogramInfo`.
#[derive(Debug, Clone, Copy)]
pub struct HistogramInfo {
    /// Corresponds to the `number_of_entries` field on `MPSImageHistogramInfo`.
    pub number_of_entries: usize,
    /// Corresponds to the `histogram_for_alpha` field on `MPSImageHistogramInfo`.
    pub histogram_for_alpha: bool,
    /// Corresponds to the `min_pixel_value` field on `MPSImageHistogramInfo`.
    pub min_pixel_value: [f32; 4],
    /// Corresponds to the `max_pixel_value` field on `MPSImageHistogramInfo`.
    pub max_pixel_value: [f32; 4],
}

opaque_handle!(ImageGaussianBlur, "Wraps `MPSImageGaussianBlur`.");
impl ImageGaussianBlur {
    /// Wraps a constructor on `MPSImageGaussianBlur`.
    #[must_use]
    pub fn new(device: &MetalDevice, sigma: f32) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_gaussian_blur_new(device.as_ptr(), sigma) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_unary_methods!(ImageGaussianBlur);

opaque_handle!(ImageBox, "Wraps `MPSImageBox`.");
impl ImageBox {
    /// Wraps a constructor on `MPSImageBox`.
    #[must_use]
    pub fn new(device: &MetalDevice, kernel_width: usize, kernel_height: usize) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_box_new(device.as_ptr(), kernel_width, kernel_height) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_unary_methods!(ImageBox);

opaque_handle!(ImageSobel, "Wraps `MPSImageSobel`.");
impl ImageSobel {
    /// Wraps a constructor on `MPSImageSobel`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_sobel_new(device.as_ptr(), core::ptr::null()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps the corresponding `MPSImageSobel` method.
    #[must_use]
    pub fn with_transform(device: &MetalDevice, transform: [f32; 3]) -> Option<Self> {
        // SAFETY: `transform` lives for the duration of the FFI call.
        let ptr = unsafe { ffi::mps_image_sobel_new(device.as_ptr(), transform.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_unary_methods!(ImageSobel);

opaque_handle!(ImageMedian, "Wraps `MPSImageMedian`.");
impl ImageMedian {
    /// Wraps a constructor on `MPSImageMedian`.
    #[must_use]
    pub fn new(device: &MetalDevice, kernel_diameter: usize) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_median_new(device.as_ptr(), kernel_diameter) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_unary_methods!(ImageMedian);

opaque_handle!(ImageConvolution, "Wraps `MPSImageConvolution`.");
impl ImageConvolution {
    /// Wraps a constructor on `MPSImageConvolution`.
    #[must_use]
    pub fn new(
        device: &MetalDevice,
        kernel_width: usize,
        kernel_height: usize,
        weights: &[f32],
    ) -> Option<Self> {
        if weights.len() != kernel_width.saturating_mul(kernel_height) {
            return None;
        }

        // SAFETY: `weights` lives for the duration of the FFI call.
        let ptr = unsafe {
            ffi::mps_image_convolution_new(
                device.as_ptr(),
                kernel_width,
                kernel_height,
                weights.as_ptr(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_unary_methods!(ImageConvolution);

opaque_handle!(ImageBilinearScale, "Wraps `MPSImageBilinearScale`.");
impl ImageBilinearScale {
    /// Wraps a constructor on `MPSImageBilinearScale`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_bilinear_scale_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Override the default fit-to-destination scale transform.
    pub fn set_scale_transform(&self, transform: ScaleTransform) {
        // SAFETY: The kernel pointer is valid for the duration of the call.
        unsafe {
            ffi::mps_image_scale_set_transform(
                self.ptr,
                transform.scale_x,
                transform.scale_y,
                transform.translate_x,
                transform.translate_y,
            );
        };
    }
}
impl_unary_methods!(ImageBilinearScale);

opaque_handle!(ImageLanczosScale, "Wraps `MPSImageLanczosScale`.");
impl ImageLanczosScale {
    /// Wraps a constructor on `MPSImageLanczosScale`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_lanczos_scale_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Override the default fit-to-destination scale transform.
    pub fn set_scale_transform(&self, transform: ScaleTransform) {
        // SAFETY: The kernel pointer is valid for the duration of the call.
        unsafe {
            ffi::mps_image_scale_set_transform(
                self.ptr,
                transform.scale_x,
                transform.scale_y,
                transform.translate_x,
                transform.translate_y,
            );
        };
    }
}
impl_unary_methods!(ImageLanczosScale);

opaque_handle!(ImageThresholdBinary, "Wraps `MPSImageThresholdBinary`.");
impl ImageThresholdBinary {
    /// Wraps a constructor on `MPSImageThresholdBinary`.
    #[must_use]
    pub fn new(device: &MetalDevice, threshold_value: f32, maximum_value: f32) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe {
            ffi::mps_image_threshold_binary_new(
                device.as_ptr(),
                threshold_value,
                maximum_value,
                core::ptr::null(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps the corresponding `MPSImageThresholdBinary` method.
    #[must_use]
    pub fn with_transform(
        device: &MetalDevice,
        threshold_value: f32,
        maximum_value: f32,
        transform: [f32; 3],
    ) -> Option<Self> {
        // SAFETY: `transform` lives for the duration of the FFI call.
        let ptr = unsafe {
            ffi::mps_image_threshold_binary_new(
                device.as_ptr(),
                threshold_value,
                maximum_value,
                transform.as_ptr(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_unary_methods!(ImageThresholdBinary);

opaque_handle!(ImageHistogram, "Wraps `MPSImageHistogram`.");
impl ImageHistogram {
    /// Wraps a constructor on `MPSImageHistogram`.
    #[must_use]
    pub fn new(device: &MetalDevice, info: HistogramInfo) -> Option<Self> {
        // SAFETY: `info` arrays live for the duration of the FFI call.
        let ptr = unsafe {
            ffi::mps_image_histogram_new(
                device.as_ptr(),
                info.number_of_entries,
                info.histogram_for_alpha,
                info.min_pixel_value.as_ptr(),
                info.max_pixel_value.as_ptr(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Encode a histogram pass using an `MPSImage` source.
    pub fn encode_image(
        &self,
        command_buffer: &CommandBuffer,
        source: &Image,
        histogram_buffer: &MetalBuffer,
        histogram_offset: usize,
    ) {
        // SAFETY: All handles come from safe wrappers and remain alive for the call.
        unsafe {
            ffi::mps_image_histogram_encode_image(
                self.ptr,
                command_buffer.as_ptr(),
                source.as_ptr(),
                histogram_buffer.as_ptr(),
                histogram_offset,
            );
        };
    }

    /// Encode a histogram pass using a raw `MTLTexture` source.
    pub fn encode_texture(
        &self,
        command_buffer: &CommandBuffer,
        source: &MetalTexture,
        histogram_buffer: &MetalBuffer,
        histogram_offset: usize,
    ) {
        // SAFETY: All handles come from safe wrappers and remain alive for the call.
        unsafe {
            ffi::mps_image_histogram_encode_texture(
                self.ptr,
                command_buffer.as_ptr(),
                source.as_ptr(),
                histogram_buffer.as_ptr(),
                histogram_offset,
            );
        };
    }

    /// Report the minimum output buffer size for the given source pixel format.
    #[must_use]
    pub fn histogram_size_for_source_format(&self, source_format: usize) -> usize {
        // SAFETY: The histogram pointer is valid for the duration of the call.
        unsafe { ffi::mps_image_histogram_size_for_source_format(self.ptr, source_format) }
    }
}

opaque_handle!(ImageStatisticsMinAndMax, "Wraps `MPSImageStatisticsMinAndMax`.");
impl ImageStatisticsMinAndMax {
    /// Wraps a constructor on `MPSImageStatisticsMinAndMax`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_statistics_min_max_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_unary_methods!(ImageStatisticsMinAndMax);

opaque_handle!(ImageStatisticsMean, "Wraps `MPSImageStatisticsMean`.");
impl ImageStatisticsMean {
    /// Wraps a constructor on `MPSImageStatisticsMean`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_statistics_mean_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_unary_methods!(ImageStatisticsMean);

opaque_handle!(ImageReduceRowMin, "Wraps `MPSImageReduceRowMin`.");
impl ImageReduceRowMin {
    /// Wraps a constructor on `MPSImageReduceRowMin`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_reduce_row_min_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_unary_methods!(ImageReduceRowMin);

opaque_handle!(ImageReduceRowMax, "Wraps `MPSImageReduceRowMax`.");
impl ImageReduceRowMax {
    /// Wraps a constructor on `MPSImageReduceRowMax`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_reduce_row_max_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_unary_methods!(ImageReduceRowMax);

opaque_handle!(ImageReduceRowMean, "Wraps `MPSImageReduceRowMean`.");
impl ImageReduceRowMean {
    /// Wraps a constructor on `MPSImageReduceRowMean`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_reduce_row_mean_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_unary_methods!(ImageReduceRowMean);

opaque_handle!(ImageReduceRowSum, "Wraps `MPSImageReduceRowSum`.");
impl ImageReduceRowSum {
    /// Wraps a constructor on `MPSImageReduceRowSum`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_reduce_row_sum_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_unary_methods!(ImageReduceRowSum);

opaque_handle!(ImageAdd, "Wraps `MPSImageAdd`.");
impl ImageAdd {
    /// Wraps a constructor on `MPSImageAdd`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        // SAFETY: `device` exposes a valid `MTLDevice` pointer.
        let ptr = unsafe { ffi::mps_image_add_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Set `primaryScale`, `secondaryScale`, and `bias` in one call.
    pub fn set_scales(&self, primary_scale: f32, secondary_scale: f32, bias: f32) {
        // SAFETY: The kernel pointer is valid for the duration of the call.
        unsafe {
            ffi::mps_image_arithmetic_set_scales_bias(
                self.ptr,
                primary_scale,
                secondary_scale,
                bias,
            );
        };
    }

    /// Clamp arithmetic results to the closed interval `[minimum_value, maximum_value]`.
    pub fn set_clamp(&self, minimum_value: f32, maximum_value: f32) {
        // SAFETY: The kernel pointer is valid for the duration of the call.
        unsafe { ffi::mps_image_arithmetic_set_clamp(self.ptr, minimum_value, maximum_value) };
    }
}
impl_binary_methods!(ImageAdd);

/// Convenience wrapper for `scale-and-add` semantics implemented with `MPSImageAdd`.
pub struct ImageScaleAndAdd {
    inner: ImageAdd,
}

impl ImageScaleAndAdd {
    /// Build an image add kernel with non-unit primary/secondary scales.
    #[must_use]
    pub fn new(
        device: &MetalDevice,
        primary_scale: f32,
        secondary_scale: f32,
        bias: f32,
    ) -> Option<Self> {
        let inner = ImageAdd::new(device)?;
        inner.set_scales(primary_scale, secondary_scale, bias);
        Some(Self { inner })
    }

    /// Wraps a Metal Performance Shaders raw value.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.inner.as_ptr()
    }

    /// Wraps the corresponding `MPSImageAdd` encode entry point.
    pub fn encode_image(
        &self,
        command_buffer: &CommandBuffer,
        primary: &Image,
        secondary: &Image,
        destination: &Image,
    ) {
        self.inner
            .encode_image(command_buffer, primary, secondary, destination);
    }

    /// Wraps the corresponding `MPSImageAdd` encode entry point.
    pub fn encode_texture(
        &self,
        command_buffer: &CommandBuffer,
        primary: &MetalTexture,
        secondary: &MetalTexture,
        destination: &MetalTexture,
    ) {
        self.inner
            .encode_texture(command_buffer, primary, secondary, destination);
    }

    /// Wraps the corresponding `MPSImageAdd` setter.
    pub fn set_primary_edge_mode(&self, edge_mode: usize) {
        self.inner.set_primary_edge_mode(edge_mode);
    }

    /// Wraps the corresponding `MPSImageAdd` setter.
    pub fn set_secondary_edge_mode(&self, edge_mode: usize) {
        self.inner.set_secondary_edge_mode(edge_mode);
    }

    /// Wraps the corresponding `MPSImageAdd` setter.
    pub fn set_clip_rect(&self, region: ImageRegion) {
        self.inner.set_clip_rect(region);
    }

    /// Wraps the corresponding `MPSImageAdd` setter.
    pub fn set_clamp(&self, minimum_value: f32, maximum_value: f32) {
        self.inner.set_clamp(minimum_value, maximum_value);
    }
}
