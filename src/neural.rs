use crate::ffi;
use crate::image::Image;
use crate::matrix::{Matrix, Vector};
use apple_metal::{CommandBuffer, MetalBuffer, MetalDevice};
use core::ffi::c_void;
use core::ptr;

/// `MPSRNNSequenceDirection` constants.
pub mod rnn_sequence_direction {
    pub const FORWARD: usize = 0;
    pub const BACKWARD: usize = 1;
}

/// `MPSRNNBidirectionalCombineMode` constants.
pub mod rnn_bidirectional_combine_mode {
    pub const NONE: usize = 0;
    pub const ADD: usize = 1;
    pub const CONCATENATE: usize = 2;
}

/// `MPSCNNConvolutionFlags` constants.
pub mod cnn_convolution_flags {
    pub const NONE: usize = 0;
}

/// `MPSCNNConvolutionWeightsLayout` constants.
pub mod cnn_convolution_weights_layout {
    pub const OHWI: u32 = 0;
}

/// `MPSNNConvolutionAccumulatorPrecisionOption` constants.
pub mod cnn_accumulator_precision_option {
    pub const HALF: usize = 0;
    pub const FLOAT: usize = 1;
}

/// `MPSNNRegularizationType` constants.
pub mod nn_regularization_type {
    pub const NONE: usize = 0;
    pub const L1: usize = 1;
    pub const L2: usize = 2;
}

pub use crate::generated::neural::*;

macro_rules! opaque_handle {
    ($name:ident) => {
        pub struct $name {
            ptr: *mut c_void,
        }

        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}

        impl Drop for $name {
            fn drop(&mut self) {
                if !self.ptr.is_null() {
                    unsafe { ffi::mps_object_release(self.ptr) };
                    self.ptr = ptr::null_mut();
                }
            }
        }

        impl $name {
            #[must_use]
            pub const fn as_ptr(&self) -> *mut c_void {
                self.ptr
            }
        }
    };
}

macro_rules! impl_filter_result_image {
    ($name:ident) => {
        impl $name {
            #[must_use]
            pub fn result_image(&self) -> Option<NNImageNode> {
                let ptr = unsafe { ffi::mps_nn_filter_node_result_image(self.ptr) };
                if ptr.is_null() {
                    None
                } else {
                    Some(NNImageNode { ptr })
                }
            }
        }
    };
}

fn retained_handle(ptr: *mut c_void) -> Option<*mut c_void> {
    let retained = unsafe { ffi::mps_object_retain(ptr) };
    if retained.is_null() {
        None
    } else {
        Some(retained)
    }
}

macro_rules! impl_rnn_descriptor_common {
    ($name:ident) => {
        impl $name {
            #[must_use]
            pub fn input_feature_channels(&self) -> usize {
                unsafe { ffi::mps_rnn_descriptor_input_feature_channels(self.ptr) }
            }

            pub fn set_input_feature_channels(&self, value: usize) {
                unsafe { ffi::mps_rnn_descriptor_set_input_feature_channels(self.ptr, value) };
            }

            #[must_use]
            pub fn output_feature_channels(&self) -> usize {
                unsafe { ffi::mps_rnn_descriptor_output_feature_channels(self.ptr) }
            }

            pub fn set_output_feature_channels(&self, value: usize) {
                unsafe { ffi::mps_rnn_descriptor_set_output_feature_channels(self.ptr, value) };
            }

            #[must_use]
            pub fn use_layer_input_unit_transform_mode(&self) -> bool {
                unsafe { ffi::mps_rnn_descriptor_use_layer_input_unit_transform_mode(self.ptr) }
            }

            pub fn set_use_layer_input_unit_transform_mode(&self, value: bool) {
                unsafe {
                    ffi::mps_rnn_descriptor_set_use_layer_input_unit_transform_mode(self.ptr, value)
                };
            }

            #[must_use]
            pub fn use_float32_weights(&self) -> bool {
                unsafe { ffi::mps_rnn_descriptor_use_float32_weights(self.ptr) }
            }

            pub fn set_use_float32_weights(&self, value: bool) {
                unsafe { ffi::mps_rnn_descriptor_set_use_float32_weights(self.ptr, value) };
            }

            #[must_use]
            pub fn layer_sequence_direction(&self) -> usize {
                unsafe { ffi::mps_rnn_descriptor_layer_sequence_direction(self.ptr) }
            }

            pub fn set_layer_sequence_direction(&self, value: usize) {
                unsafe { ffi::mps_rnn_descriptor_set_layer_sequence_direction(self.ptr, value) };
            }
        }
    };
}

macro_rules! impl_optimizer_common {
    ($name:ident) => {
        impl $name {
            #[must_use]
            pub fn learning_rate(&self) -> f32 {
                unsafe { ffi::mps_nn_optimizer_learning_rate(self.ptr) }
            }

            pub fn set_learning_rate(&self, value: f32) {
                unsafe { ffi::mps_nn_optimizer_set_learning_rate(self.ptr, value) };
            }

            #[must_use]
            pub fn gradient_rescale(&self) -> f32 {
                unsafe { ffi::mps_nn_optimizer_gradient_rescale(self.ptr) }
            }

            #[must_use]
            pub fn apply_gradient_clipping(&self) -> bool {
                unsafe { ffi::mps_nn_optimizer_apply_gradient_clipping(self.ptr) }
            }

            pub fn set_apply_gradient_clipping(&self, value: bool) {
                unsafe { ffi::mps_nn_optimizer_set_apply_gradient_clipping(self.ptr, value) };
            }

            #[must_use]
            pub fn gradient_clip_max(&self) -> f32 {
                unsafe { ffi::mps_nn_optimizer_gradient_clip_max(self.ptr) }
            }

            #[must_use]
            pub fn gradient_clip_min(&self) -> f32 {
                unsafe { ffi::mps_nn_optimizer_gradient_clip_min(self.ptr) }
            }

            #[must_use]
            pub fn regularization_scale(&self) -> f32 {
                unsafe { ffi::mps_nn_optimizer_regularization_scale(self.ptr) }
            }

            #[must_use]
            pub fn regularization_type(&self) -> usize {
                unsafe { ffi::mps_nn_optimizer_regularization_type(self.ptr) }
            }
        }
    };
}

opaque_handle!(NNImageNode);
impl NNImageNode {
    #[must_use]
    pub fn new() -> Option<Self> {
        let ptr = unsafe { ffi::mps_nn_image_node_new() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn exported() -> Option<Self> {
        let ptr = unsafe { ffi::mps_nn_image_node_exported() };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn format(&self) -> usize {
        unsafe { ffi::mps_nn_image_node_format(self.ptr) }
    }

    pub fn set_format(&self, format: usize) {
        unsafe { ffi::mps_nn_image_node_set_format(self.ptr, format) };
    }

    #[must_use]
    pub fn export_from_graph(&self) -> bool {
        unsafe { ffi::mps_nn_image_node_export_from_graph(self.ptr) }
    }

    pub fn set_export_from_graph(&self, export: bool) {
        unsafe { ffi::mps_nn_image_node_set_export_from_graph(self.ptr, export) };
    }

    #[must_use]
    pub fn synchronize_resource(&self) -> bool {
        unsafe { ffi::mps_nn_image_node_synchronize_resource(self.ptr) }
    }

    pub fn set_synchronize_resource(&self, synchronize: bool) {
        unsafe { ffi::mps_nn_image_node_set_synchronize_resource(self.ptr, synchronize) };
    }

    pub fn use_default_allocator(&self) {
        unsafe { ffi::mps_nn_image_node_use_default_allocator(self.ptr) };
    }
}

opaque_handle!(CnnNeuronReluNode);
impl CnnNeuronReluNode {
    #[must_use]
    pub fn new(source: &NNImageNode, a: f32) -> Option<Self> {
        let ptr = unsafe { ffi::mps_cnn_neuron_relu_node_new(source.as_ptr(), a) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_filter_result_image!(CnnNeuronReluNode);

opaque_handle!(CnnPoolingMaxNode);
impl CnnPoolingMaxNode {
    #[must_use]
    pub fn new(source: &NNImageNode, filter_size: usize, stride: usize) -> Option<Self> {
        let ptr =
            unsafe { ffi::mps_cnn_pooling_max_node_new(source.as_ptr(), filter_size, stride) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_filter_result_image!(CnnPoolingMaxNode);

opaque_handle!(CnnSoftMaxNode);
impl CnnSoftMaxNode {
    #[must_use]
    pub fn new(source: &NNImageNode) -> Option<Self> {
        let ptr = unsafe { ffi::mps_cnn_softmax_node_new(source.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_filter_result_image!(CnnSoftMaxNode);

opaque_handle!(CnnUpsamplingNearestNode);
impl CnnUpsamplingNearestNode {
    #[must_use]
    pub fn new(source: &NNImageNode, scale_x: usize, scale_y: usize) -> Option<Self> {
        let ptr =
            unsafe { ffi::mps_cnn_upsampling_nearest_node_new(source.as_ptr(), scale_x, scale_y) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }
}
impl_filter_result_image!(CnnUpsamplingNearestNode);

opaque_handle!(NNGraph);
impl NNGraph {
    #[must_use]
    pub fn new(
        device: &MetalDevice,
        result_image: &NNImageNode,
        result_image_is_needed: bool,
    ) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_nn_graph_new(
                device.as_ptr(),
                result_image.as_ptr(),
                result_image_is_needed,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn source_image_count(&self) -> usize {
        unsafe { ffi::mps_nn_graph_source_image_count(self.ptr) }
    }

    #[must_use]
    pub fn format(&self) -> usize {
        unsafe { ffi::mps_nn_graph_format(self.ptr) }
    }

    pub fn set_format(&self, format: usize) {
        unsafe { ffi::mps_nn_graph_set_format(self.ptr, format) };
    }

    pub fn set_output_state_is_temporary(&self, temporary: bool) {
        unsafe { ffi::mps_nn_graph_set_output_state_is_temporary(self.ptr, temporary) };
    }

    pub fn use_default_destination_image_allocator(&self) {
        unsafe { ffi::mps_nn_graph_use_default_destination_image_allocator(self.ptr) };
    }

    pub fn reload_from_data_sources(&self) {
        unsafe { ffi::mps_nn_graph_reload_from_data_sources(self.ptr) };
    }

    #[must_use]
    pub fn encode(
        &self,
        command_buffer: &CommandBuffer,
        source_images: &[&Image],
    ) -> Option<Image> {
        let handles: Vec<_> = source_images.iter().map(|image| image.as_ptr()).collect();
        let source_handles = if handles.is_empty() {
            ptr::null()
        } else {
            handles.as_ptr()
        };
        let ptr = unsafe {
            ffi::mps_nn_graph_encode(
                self.ptr,
                command_buffer.as_ptr(),
                source_images.len(),
                source_handles,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { Image::from_raw(ptr) })
        }
    }
}

opaque_handle!(CnnConvolutionDescriptor);
impl CnnConvolutionDescriptor {
    #[must_use]
    pub fn new(
        kernel_width: usize,
        kernel_height: usize,
        input_feature_channels: usize,
        output_feature_channels: usize,
    ) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_cnn_convolution_descriptor_new(
                kernel_width,
                kernel_height,
                input_feature_channels,
                output_feature_channels,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn kernel_width(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_descriptor_kernel_width(self.ptr) }
    }

    #[must_use]
    pub fn kernel_height(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_descriptor_kernel_height(self.ptr) }
    }

    #[must_use]
    pub fn input_feature_channels(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_descriptor_input_feature_channels(self.ptr) }
    }

    #[must_use]
    pub fn output_feature_channels(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_descriptor_output_feature_channels(self.ptr) }
    }

    #[must_use]
    pub fn stride_in_pixels_x(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_descriptor_stride_in_pixels_x(self.ptr) }
    }

    pub fn set_stride_in_pixels_x(&self, value: usize) {
        unsafe { ffi::mps_cnn_convolution_descriptor_set_stride_in_pixels_x(self.ptr, value) };
    }

    #[must_use]
    pub fn stride_in_pixels_y(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_descriptor_stride_in_pixels_y(self.ptr) }
    }

    pub fn set_stride_in_pixels_y(&self, value: usize) {
        unsafe { ffi::mps_cnn_convolution_descriptor_set_stride_in_pixels_y(self.ptr, value) };
    }

    #[must_use]
    pub fn groups(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_descriptor_groups(self.ptr) }
    }

    pub fn set_groups(&self, value: usize) {
        unsafe { ffi::mps_cnn_convolution_descriptor_set_groups(self.ptr, value) };
    }

    #[must_use]
    pub fn dilation_rate_x(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_descriptor_dilation_rate_x(self.ptr) }
    }

    pub fn set_dilation_rate_x(&self, value: usize) {
        unsafe { ffi::mps_cnn_convolution_descriptor_set_dilation_rate_x(self.ptr, value) };
    }

    #[must_use]
    pub fn dilation_rate_y(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_descriptor_dilation_rate_y(self.ptr) }
    }

    pub fn set_dilation_rate_y(&self, value: usize) {
        unsafe { ffi::mps_cnn_convolution_descriptor_set_dilation_rate_y(self.ptr, value) };
    }
}

opaque_handle!(RnnSingleGateDescriptor);
impl RnnSingleGateDescriptor {
    #[must_use]
    pub fn new(input_feature_channels: usize, output_feature_channels: usize) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_rnn_single_gate_descriptor_new(input_feature_channels, output_feature_channels)
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn input_feature_channels(&self) -> usize {
        unsafe { ffi::mps_rnn_single_gate_descriptor_input_feature_channels(self.ptr) }
    }

    pub fn set_input_feature_channels(&self, value: usize) {
        unsafe { ffi::mps_rnn_single_gate_descriptor_set_input_feature_channels(self.ptr, value) };
    }

    #[must_use]
    pub fn output_feature_channels(&self) -> usize {
        unsafe { ffi::mps_rnn_single_gate_descriptor_output_feature_channels(self.ptr) }
    }

    pub fn set_output_feature_channels(&self, value: usize) {
        unsafe { ffi::mps_rnn_single_gate_descriptor_set_output_feature_channels(self.ptr, value) };
    }

    #[must_use]
    pub fn use_layer_input_unit_transform_mode(&self) -> bool {
        unsafe { ffi::mps_rnn_single_gate_descriptor_use_layer_input_unit_transform_mode(self.ptr) }
    }

    pub fn set_use_layer_input_unit_transform_mode(&self, value: bool) {
        unsafe {
            ffi::mps_rnn_single_gate_descriptor_set_use_layer_input_unit_transform_mode(
                self.ptr, value,
            );
        };
    }

    #[must_use]
    pub fn use_float32_weights(&self) -> bool {
        unsafe { ffi::mps_rnn_single_gate_descriptor_use_float32_weights(self.ptr) }
    }

    pub fn set_use_float32_weights(&self, value: bool) {
        unsafe { ffi::mps_rnn_single_gate_descriptor_set_use_float32_weights(self.ptr, value) };
    }

    #[must_use]
    pub fn layer_sequence_direction(&self) -> usize {
        unsafe { ffi::mps_rnn_single_gate_descriptor_layer_sequence_direction(self.ptr) }
    }

    pub fn set_layer_sequence_direction(&self, value: usize) {
        unsafe {
            ffi::mps_rnn_single_gate_descriptor_set_layer_sequence_direction(self.ptr, value);
        };
    }

    #[must_use]
    pub fn as_descriptor(&self) -> Option<RnnDescriptor> {
        retained_handle(self.ptr).map(|ptr| RnnDescriptor { ptr })
    }
}

opaque_handle!(CnnConvolution);
impl CnnConvolution {
    #[must_use]
    pub fn new(
        device: &MetalDevice,
        descriptor: &CnnConvolutionDescriptor,
        kernel_weights: &[f32],
        bias_terms: Option<&[f32]>,
        flags: usize,
    ) -> Option<Self> {
        if kernel_weights.is_empty() {
            return None;
        }
        let bias_terms_ptr = bias_terms.map_or(ptr::null(), <[f32]>::as_ptr);
        let ptr = unsafe {
            ffi::mps_cnn_convolution_new(
                device.as_ptr(),
                descriptor.as_ptr(),
                kernel_weights.as_ptr(),
                bias_terms_ptr,
                flags,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn input_feature_channels(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_input_feature_channels(self.ptr) }
    }

    #[must_use]
    pub fn output_feature_channels(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_output_feature_channels(self.ptr) }
    }

    #[must_use]
    pub fn groups(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_groups(self.ptr) }
    }

    #[must_use]
    pub fn sub_pixel_scale_factor(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_sub_pixel_scale_factor(self.ptr) }
    }

    #[must_use]
    pub fn channel_multiplier(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_channel_multiplier(self.ptr) }
    }

    #[must_use]
    pub fn accumulator_precision_option(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_accumulator_precision_option(self.ptr) }
    }

    pub fn set_accumulator_precision_option(&self, value: usize) {
        unsafe { ffi::mps_cnn_convolution_set_accumulator_precision_option(self.ptr, value) };
    }

    pub fn encode_image(
        &self,
        command_buffer: &CommandBuffer,
        source: &Image,
        destination: &Image,
    ) {
        unsafe {
            ffi::mps_cnn_convolution_encode_image(
                self.ptr,
                command_buffer.as_ptr(),
                source.as_ptr(),
                destination.as_ptr(),
            );
        };
    }
}

opaque_handle!(CnnConvolutionWeightsAndBiasesState);
impl CnnConvolutionWeightsAndBiasesState {
    #[must_use]
    pub fn new_with_buffers(weights: &MetalBuffer, biases: Option<&MetalBuffer>) -> Option<Self> {
        let biases_ptr = biases.map_or(ptr::null_mut(), MetalBuffer::as_ptr);
        let ptr = unsafe {
            ffi::mps_cnn_convolution_weights_and_biases_state_new(weights.as_ptr(), biases_ptr)
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn new_with_offsets(
        weights: &MetalBuffer,
        weights_offset: usize,
        biases: Option<&MetalBuffer>,
        biases_offset: usize,
        descriptor: &CnnConvolutionDescriptor,
    ) -> Option<Self> {
        let biases_ptr = biases.map_or(ptr::null_mut(), MetalBuffer::as_ptr);
        let ptr = unsafe {
            ffi::mps_cnn_convolution_weights_and_biases_state_new_with_offsets(
                weights.as_ptr(),
                weights_offset,
                biases_ptr,
                biases_offset,
                descriptor.as_ptr(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn new_with_device(
        device: &MetalDevice,
        descriptor: &CnnConvolutionDescriptor,
    ) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_cnn_convolution_weights_and_biases_state_new_with_device(
                device.as_ptr(),
                descriptor.as_ptr(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn weights_offset(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_weights_and_biases_state_weights_offset(self.ptr) }
    }

    #[must_use]
    pub fn biases_offset(&self) -> usize {
        unsafe { ffi::mps_cnn_convolution_weights_and_biases_state_biases_offset(self.ptr) }
    }
}

opaque_handle!(NNOptimizerDescriptor);
impl NNOptimizerDescriptor {
    #[must_use]
    pub fn new(
        learning_rate: f32,
        gradient_rescale: f32,
        regularization_type: usize,
        regularization_scale: f32,
    ) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_nn_optimizer_descriptor_new(
                learning_rate,
                gradient_rescale,
                regularization_type,
                regularization_scale,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn with_gradient_clipping(
        learning_rate: f32,
        gradient_rescale: f32,
        apply_gradient_clipping: bool,
        gradient_clip_max: f32,
        gradient_clip_min: f32,
        regularization_type: usize,
        regularization_scale: f32,
    ) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_nn_optimizer_descriptor_new_with_gradient_clipping(
                learning_rate,
                gradient_rescale,
                apply_gradient_clipping,
                gradient_clip_max,
                gradient_clip_min,
                regularization_type,
                regularization_scale,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn learning_rate(&self) -> f32 {
        unsafe { ffi::mps_nn_optimizer_descriptor_learning_rate(self.ptr) }
    }

    pub fn set_learning_rate(&self, value: f32) {
        unsafe { ffi::mps_nn_optimizer_descriptor_set_learning_rate(self.ptr, value) };
    }

    #[must_use]
    pub fn gradient_rescale(&self) -> f32 {
        unsafe { ffi::mps_nn_optimizer_descriptor_gradient_rescale(self.ptr) }
    }

    pub fn set_gradient_rescale(&self, value: f32) {
        unsafe { ffi::mps_nn_optimizer_descriptor_set_gradient_rescale(self.ptr, value) };
    }

    #[must_use]
    pub fn apply_gradient_clipping(&self) -> bool {
        unsafe { ffi::mps_nn_optimizer_descriptor_apply_gradient_clipping(self.ptr) }
    }

    pub fn set_apply_gradient_clipping(&self, value: bool) {
        unsafe { ffi::mps_nn_optimizer_descriptor_set_apply_gradient_clipping(self.ptr, value) };
    }

    #[must_use]
    pub fn gradient_clip_max(&self) -> f32 {
        unsafe { ffi::mps_nn_optimizer_descriptor_gradient_clip_max(self.ptr) }
    }

    pub fn set_gradient_clip_max(&self, value: f32) {
        unsafe { ffi::mps_nn_optimizer_descriptor_set_gradient_clip_max(self.ptr, value) };
    }

    #[must_use]
    pub fn gradient_clip_min(&self) -> f32 {
        unsafe { ffi::mps_nn_optimizer_descriptor_gradient_clip_min(self.ptr) }
    }

    pub fn set_gradient_clip_min(&self, value: f32) {
        unsafe { ffi::mps_nn_optimizer_descriptor_set_gradient_clip_min(self.ptr, value) };
    }

    #[must_use]
    pub fn regularization_scale(&self) -> f32 {
        unsafe { ffi::mps_nn_optimizer_descriptor_regularization_scale(self.ptr) }
    }

    pub fn set_regularization_scale(&self, value: f32) {
        unsafe { ffi::mps_nn_optimizer_descriptor_set_regularization_scale(self.ptr, value) };
    }

    #[must_use]
    pub fn regularization_type(&self) -> usize {
        unsafe { ffi::mps_nn_optimizer_descriptor_regularization_type(self.ptr) }
    }

    pub fn set_regularization_type(&self, value: usize) {
        unsafe { ffi::mps_nn_optimizer_descriptor_set_regularization_type(self.ptr, value) };
    }
}

opaque_handle!(NNOptimizer);
impl_optimizer_common!(NNOptimizer);

opaque_handle!(NNOptimizerStochasticGradientDescent);
impl_optimizer_common!(NNOptimizerStochasticGradientDescent);
impl NNOptimizerStochasticGradientDescent {
    #[must_use]
    pub fn new(device: &MetalDevice, learning_rate: f32) -> Option<Self> {
        let ptr = unsafe { ffi::mps_nn_optimizer_sgd_new(device.as_ptr(), learning_rate) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn new_with_options(
        device: &MetalDevice,
        momentum_scale: f32,
        use_nesterov_momentum: bool,
        optimizer_descriptor: &NNOptimizerDescriptor,
    ) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_nn_optimizer_sgd_new_with_options(
                device.as_ptr(),
                momentum_scale,
                use_nesterov_momentum,
                optimizer_descriptor.as_ptr(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn as_optimizer(&self) -> Option<NNOptimizer> {
        retained_handle(self.ptr).map(|ptr| NNOptimizer { ptr })
    }

    #[must_use]
    pub fn momentum_scale(&self) -> f32 {
        unsafe { ffi::mps_nn_optimizer_sgd_momentum_scale(self.ptr) }
    }

    #[must_use]
    pub fn use_nesterov_momentum(&self) -> bool {
        unsafe { ffi::mps_nn_optimizer_sgd_use_nesterov_momentum(self.ptr) }
    }

    pub fn encode_vector(
        &self,
        command_buffer: &CommandBuffer,
        input_gradient_vector: &Vector,
        input_values_vector: &Vector,
        input_momentum_vector: Option<&Vector>,
        result_values_vector: &Vector,
    ) {
        let input_momentum_ptr = input_momentum_vector.map_or(ptr::null_mut(), Vector::as_ptr);
        unsafe {
            ffi::mps_nn_optimizer_sgd_encode_vector(
                self.ptr,
                command_buffer.as_ptr(),
                input_gradient_vector.as_ptr(),
                input_values_vector.as_ptr(),
                input_momentum_ptr,
                result_values_vector.as_ptr(),
            );
        };
    }

    pub fn encode_matrix(
        &self,
        command_buffer: &CommandBuffer,
        input_gradient_matrix: &Matrix,
        input_values_matrix: &Matrix,
        input_momentum_matrix: Option<&Matrix>,
        result_values_matrix: &Matrix,
    ) {
        let input_momentum_ptr = input_momentum_matrix.map_or(ptr::null_mut(), Matrix::as_ptr);
        unsafe {
            ffi::mps_nn_optimizer_sgd_encode_matrix(
                self.ptr,
                command_buffer.as_ptr(),
                input_gradient_matrix.as_ptr(),
                input_values_matrix.as_ptr(),
                input_momentum_ptr,
                result_values_matrix.as_ptr(),
            );
        };
    }
}

opaque_handle!(NNOptimizerRmsProp);
impl_optimizer_common!(NNOptimizerRmsProp);
impl NNOptimizerRmsProp {
    #[must_use]
    pub fn new(device: &MetalDevice, learning_rate: f32) -> Option<Self> {
        let ptr = unsafe { ffi::mps_nn_optimizer_rmsprop_new(device.as_ptr(), learning_rate) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn new_with_options(
        device: &MetalDevice,
        decay: f64,
        epsilon: f32,
        optimizer_descriptor: &NNOptimizerDescriptor,
    ) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_nn_optimizer_rmsprop_new_with_options(
                device.as_ptr(),
                decay,
                epsilon,
                optimizer_descriptor.as_ptr(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn as_optimizer(&self) -> Option<NNOptimizer> {
        retained_handle(self.ptr).map(|ptr| NNOptimizer { ptr })
    }

    #[must_use]
    pub fn decay(&self) -> f64 {
        unsafe { ffi::mps_nn_optimizer_rmsprop_decay(self.ptr) }
    }

    #[must_use]
    pub fn epsilon(&self) -> f32 {
        unsafe { ffi::mps_nn_optimizer_rmsprop_epsilon(self.ptr) }
    }

    pub fn encode_vector(
        &self,
        command_buffer: &CommandBuffer,
        input_gradient_vector: &Vector,
        input_values_vector: &Vector,
        input_sum_of_squares_vector: &Vector,
        result_values_vector: &Vector,
    ) {
        unsafe {
            ffi::mps_nn_optimizer_rmsprop_encode_vector(
                self.ptr,
                command_buffer.as_ptr(),
                input_gradient_vector.as_ptr(),
                input_values_vector.as_ptr(),
                input_sum_of_squares_vector.as_ptr(),
                result_values_vector.as_ptr(),
            );
        };
    }

    pub fn encode_matrix(
        &self,
        command_buffer: &CommandBuffer,
        input_gradient_matrix: &Matrix,
        input_values_matrix: &Matrix,
        input_sum_of_squares_matrix: &Matrix,
        result_values_matrix: &Matrix,
    ) {
        unsafe {
            ffi::mps_nn_optimizer_rmsprop_encode_matrix(
                self.ptr,
                command_buffer.as_ptr(),
                input_gradient_matrix.as_ptr(),
                input_values_matrix.as_ptr(),
                input_sum_of_squares_matrix.as_ptr(),
                result_values_matrix.as_ptr(),
            );
        };
    }
}

opaque_handle!(NNOptimizerAdam);
impl_optimizer_common!(NNOptimizerAdam);
impl NNOptimizerAdam {
    #[must_use]
    pub fn new(device: &MetalDevice, learning_rate: f32) -> Option<Self> {
        let ptr = unsafe { ffi::mps_nn_optimizer_adam_new(device.as_ptr(), learning_rate) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn new_with_options(
        device: &MetalDevice,
        beta1: f64,
        beta2: f64,
        epsilon: f32,
        time_step: usize,
        optimizer_descriptor: &NNOptimizerDescriptor,
    ) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_nn_optimizer_adam_new_with_options(
                device.as_ptr(),
                beta1,
                beta2,
                epsilon,
                time_step,
                optimizer_descriptor.as_ptr(),
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn as_optimizer(&self) -> Option<NNOptimizer> {
        retained_handle(self.ptr).map(|ptr| NNOptimizer { ptr })
    }

    #[must_use]
    pub fn beta1(&self) -> f64 {
        unsafe { ffi::mps_nn_optimizer_adam_beta1(self.ptr) }
    }

    #[must_use]
    pub fn beta2(&self) -> f64 {
        unsafe { ffi::mps_nn_optimizer_adam_beta2(self.ptr) }
    }

    #[must_use]
    pub fn epsilon(&self) -> f32 {
        unsafe { ffi::mps_nn_optimizer_adam_epsilon(self.ptr) }
    }

    #[must_use]
    pub fn time_step(&self) -> usize {
        unsafe { ffi::mps_nn_optimizer_adam_time_step(self.ptr) }
    }

    pub fn set_time_step(&self, value: usize) {
        unsafe { ffi::mps_nn_optimizer_adam_set_time_step(self.ptr, value) };
    }

    pub fn encode_vector(
        &self,
        command_buffer: &CommandBuffer,
        input_gradient_vector: &Vector,
        input_values_vector: &Vector,
        input_momentum_vector: &Vector,
        input_velocity_vector: &Vector,
        result_values_vector: &Vector,
    ) {
        unsafe {
            ffi::mps_nn_optimizer_adam_encode_vector(
                self.ptr,
                command_buffer.as_ptr(),
                input_gradient_vector.as_ptr(),
                input_values_vector.as_ptr(),
                input_momentum_vector.as_ptr(),
                input_velocity_vector.as_ptr(),
                result_values_vector.as_ptr(),
            );
        };
    }

    pub fn encode_matrix(
        &self,
        command_buffer: &CommandBuffer,
        input_gradient_matrix: &Matrix,
        input_values_matrix: &Matrix,
        input_momentum_matrix: &Matrix,
        input_velocity_matrix: &Matrix,
        result_values_matrix: &Matrix,
    ) {
        unsafe {
            ffi::mps_nn_optimizer_adam_encode_matrix(
                self.ptr,
                command_buffer.as_ptr(),
                input_gradient_matrix.as_ptr(),
                input_values_matrix.as_ptr(),
                input_momentum_matrix.as_ptr(),
                input_velocity_matrix.as_ptr(),
                result_values_matrix.as_ptr(),
            );
        };
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_amsgrad_vector(
        &self,
        command_buffer: &CommandBuffer,
        input_gradient_vector: &Vector,
        input_values_vector: &Vector,
        input_momentum_vector: &Vector,
        input_velocity_vector: &Vector,
        maximum_velocity_vector: Option<&Vector>,
        result_values_vector: &Vector,
    ) {
        let maximum_velocity_ptr = maximum_velocity_vector.map_or(ptr::null_mut(), Vector::as_ptr);
        unsafe {
            ffi::mps_nn_optimizer_adam_encode_amsgrad_vector(
                self.ptr,
                command_buffer.as_ptr(),
                input_gradient_vector.as_ptr(),
                input_values_vector.as_ptr(),
                input_momentum_vector.as_ptr(),
                input_velocity_vector.as_ptr(),
                maximum_velocity_ptr,
                result_values_vector.as_ptr(),
            );
        };
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_amsgrad_matrix(
        &self,
        command_buffer: &CommandBuffer,
        input_gradient_matrix: &Matrix,
        input_values_matrix: &Matrix,
        input_momentum_matrix: &Matrix,
        input_velocity_matrix: &Matrix,
        maximum_velocity_matrix: Option<&Matrix>,
        result_values_matrix: &Matrix,
    ) {
        let maximum_velocity_ptr = maximum_velocity_matrix.map_or(ptr::null_mut(), Matrix::as_ptr);
        unsafe {
            ffi::mps_nn_optimizer_adam_encode_amsgrad_matrix(
                self.ptr,
                command_buffer.as_ptr(),
                input_gradient_matrix.as_ptr(),
                input_values_matrix.as_ptr(),
                input_momentum_matrix.as_ptr(),
                input_velocity_matrix.as_ptr(),
                maximum_velocity_ptr,
                result_values_matrix.as_ptr(),
            );
        };
    }
}

opaque_handle!(RnnDescriptor);
impl_rnn_descriptor_common!(RnnDescriptor);

opaque_handle!(GruDescriptor);
impl_rnn_descriptor_common!(GruDescriptor);
impl GruDescriptor {
    #[must_use]
    pub fn new(input_feature_channels: usize, output_feature_channels: usize) -> Option<Self> {
        let ptr =
            unsafe { ffi::mps_gru_descriptor_new(input_feature_channels, output_feature_channels) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn as_descriptor(&self) -> Option<RnnDescriptor> {
        retained_handle(self.ptr).map(|ptr| RnnDescriptor { ptr })
    }

    #[must_use]
    pub fn gate_pnorm_value(&self) -> f32 {
        unsafe { ffi::mps_gru_descriptor_gate_pnorm_value(self.ptr) }
    }

    pub fn set_gate_pnorm_value(&self, value: f32) {
        unsafe { ffi::mps_gru_descriptor_set_gate_pnorm_value(self.ptr, value) };
    }

    #[must_use]
    pub fn flip_output_gates(&self) -> bool {
        unsafe { ffi::mps_gru_descriptor_flip_output_gates(self.ptr) }
    }

    pub fn set_flip_output_gates(&self, value: bool) {
        unsafe { ffi::mps_gru_descriptor_set_flip_output_gates(self.ptr, value) };
    }
}

opaque_handle!(LstmDescriptor);
impl_rnn_descriptor_common!(LstmDescriptor);
impl LstmDescriptor {
    #[must_use]
    pub fn new(input_feature_channels: usize, output_feature_channels: usize) -> Option<Self> {
        let ptr = unsafe {
            ffi::mps_lstm_descriptor_new(input_feature_channels, output_feature_channels)
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn as_descriptor(&self) -> Option<RnnDescriptor> {
        retained_handle(self.ptr).map(|ptr| RnnDescriptor { ptr })
    }

    #[must_use]
    pub fn memory_weights_are_diagonal(&self) -> bool {
        unsafe { ffi::mps_lstm_descriptor_memory_weights_are_diagonal(self.ptr) }
    }

    pub fn set_memory_weights_are_diagonal(&self, value: bool) {
        unsafe { ffi::mps_lstm_descriptor_set_memory_weights_are_diagonal(self.ptr, value) };
    }

    #[must_use]
    pub fn cell_to_output_neuron_type(&self) -> usize {
        unsafe { ffi::mps_lstm_descriptor_cell_to_output_neuron_type(self.ptr) }
    }

    pub fn set_cell_to_output_neuron_type(&self, value: usize) {
        unsafe { ffi::mps_lstm_descriptor_set_cell_to_output_neuron_type(self.ptr, value) };
    }

    #[must_use]
    pub fn cell_to_output_neuron_param_a(&self) -> f32 {
        unsafe { ffi::mps_lstm_descriptor_cell_to_output_neuron_param_a(self.ptr) }
    }

    pub fn set_cell_to_output_neuron_param_a(&self, value: f32) {
        unsafe { ffi::mps_lstm_descriptor_set_cell_to_output_neuron_param_a(self.ptr, value) };
    }

    #[must_use]
    pub fn cell_to_output_neuron_param_b(&self) -> f32 {
        unsafe { ffi::mps_lstm_descriptor_cell_to_output_neuron_param_b(self.ptr) }
    }

    pub fn set_cell_to_output_neuron_param_b(&self, value: f32) {
        unsafe { ffi::mps_lstm_descriptor_set_cell_to_output_neuron_param_b(self.ptr, value) };
    }

    #[must_use]
    pub fn cell_to_output_neuron_param_c(&self) -> f32 {
        unsafe { ffi::mps_lstm_descriptor_cell_to_output_neuron_param_c(self.ptr) }
    }

    pub fn set_cell_to_output_neuron_param_c(&self, value: f32) {
        unsafe { ffi::mps_lstm_descriptor_set_cell_to_output_neuron_param_c(self.ptr, value) };
    }
}

opaque_handle!(RnnRecurrentImageState);
impl RnnRecurrentImageState {
    #[must_use]
    pub fn recurrent_output_image_for_layer_index(&self, layer_index: usize) -> Option<Image> {
        let ptr = unsafe {
            ffi::mps_rnn_recurrent_image_state_recurrent_output_image(self.ptr, layer_index)
        };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { Image::from_raw(ptr) })
        }
    }

    #[must_use]
    pub fn memory_cell_image_for_layer_index(&self, layer_index: usize) -> Option<Image> {
        let ptr =
            unsafe { ffi::mps_rnn_recurrent_image_state_memory_cell_image(self.ptr, layer_index) };
        if ptr.is_null() {
            None
        } else {
            Some(unsafe { Image::from_raw(ptr) })
        }
    }
}

opaque_handle!(RnnImageInferenceLayer);
impl RnnImageInferenceLayer {
    #[must_use]
    pub fn new(device: &MetalDevice, descriptor: &RnnDescriptor) -> Option<Self> {
        let ptr =
            unsafe { ffi::mps_rnn_image_inference_layer_new(device.as_ptr(), descriptor.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn new_stack(device: &MetalDevice, descriptors: &[&RnnDescriptor]) -> Option<Self> {
        let handles: Vec<_> = descriptors
            .iter()
            .map(|descriptor| descriptor.as_ptr())
            .collect();
        let handles_ptr = if handles.is_empty() {
            ptr::null()
        } else {
            handles.as_ptr()
        };
        let ptr = unsafe {
            ffi::mps_rnn_image_inference_layer_new_stack(
                device.as_ptr(),
                descriptors.len(),
                handles_ptr,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    #[must_use]
    pub fn input_feature_channels(&self) -> usize {
        unsafe { ffi::mps_rnn_image_inference_layer_input_feature_channels(self.ptr) }
    }

    #[must_use]
    pub fn output_feature_channels(&self) -> usize {
        unsafe { ffi::mps_rnn_image_inference_layer_output_feature_channels(self.ptr) }
    }

    #[must_use]
    pub fn number_of_layers(&self) -> usize {
        unsafe { ffi::mps_rnn_image_inference_layer_number_of_layers(self.ptr) }
    }

    #[must_use]
    pub fn recurrent_output_is_temporary(&self) -> bool {
        unsafe { ffi::mps_rnn_image_inference_layer_recurrent_output_is_temporary(self.ptr) }
    }

    pub fn set_recurrent_output_is_temporary(&self, value: bool) {
        unsafe {
            ffi::mps_rnn_image_inference_layer_set_recurrent_output_is_temporary(self.ptr, value);
        }
    }

    #[must_use]
    pub fn store_all_intermediate_states(&self) -> bool {
        unsafe { ffi::mps_rnn_image_inference_layer_store_all_intermediate_states(self.ptr) }
    }

    pub fn set_store_all_intermediate_states(&self, value: bool) {
        unsafe {
            ffi::mps_rnn_image_inference_layer_set_store_all_intermediate_states(self.ptr, value);
        }
    }

    #[must_use]
    pub fn bidirectional_combine_mode(&self) -> usize {
        unsafe { ffi::mps_rnn_image_inference_layer_bidirectional_combine_mode(self.ptr) }
    }

    pub fn set_bidirectional_combine_mode(&self, value: usize) {
        unsafe {
            ffi::mps_rnn_image_inference_layer_set_bidirectional_combine_mode(self.ptr, value);
        }
    }

    #[must_use]
    pub fn encode_sequence(
        &self,
        command_buffer: &CommandBuffer,
        source_images: &[&Image],
        destination_images: &[&Image],
        recurrent_input_state: Option<&RnnRecurrentImageState>,
    ) -> Option<RnnRecurrentImageState> {
        if source_images.len() != destination_images.len() {
            return None;
        }
        let source_handles: Vec<_> = source_images.iter().map(|image| image.as_ptr()).collect();
        let destination_handles: Vec<_> = destination_images
            .iter()
            .map(|image| image.as_ptr())
            .collect();
        let source_ptr = if source_handles.is_empty() {
            ptr::null()
        } else {
            source_handles.as_ptr()
        };
        let destination_ptr = if destination_handles.is_empty() {
            ptr::null()
        } else {
            destination_handles.as_ptr()
        };
        let recurrent_input_ptr =
            recurrent_input_state.map_or(ptr::null_mut(), RnnRecurrentImageState::as_ptr);
        let ptr = unsafe {
            ffi::mps_rnn_image_inference_layer_encode_sequence(
                self.ptr,
                command_buffer.as_ptr(),
                source_images.len(),
                source_ptr,
                destination_ptr,
                recurrent_input_ptr,
            )
        };
        if ptr.is_null() {
            None
        } else {
            Some(RnnRecurrentImageState { ptr })
        }
    }
}
