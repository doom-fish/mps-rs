use crate::ffi;
use crate::image::Image;
use apple_metal::{CommandBuffer, MetalDevice};
use core::ffi::c_void;
use core::ptr;

/// `MPSRNNSequenceDirection` constants.
pub mod rnn_sequence_direction {
    pub const FORWARD: usize = 0;
    pub const BACKWARD: usize = 1;
}

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
}
