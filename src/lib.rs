#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::doc_markdown)]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

/// Metal Performance Shaders core device and command-buffer wrappers.
pub mod core;
/// Error types returned by the safe Metal Performance Shaders wrappers.
pub mod error;
#[doc(hidden)]
pub mod ffi;
/// Metal Performance Shaders image-filter wrappers.
pub mod filters;
mod generated;
/// Metal Performance Shaders image wrappers and related raw-value helpers.
pub mod image;
/// Metal Performance Shaders matrix and vector wrappers.
pub mod matrix;
/// Metal Performance Shaders NDArray wrappers.
pub mod ndarray;
/// Metal Performance Shaders neural-network graph and optimizer wrappers.
pub mod neural;
/// Metal Performance Shaders ray-tracing wrappers.
pub mod ray;
/// Metal Performance Shaders state wrappers.
pub mod state;

pub use crate::core::{
    device_options, hint_temporary_memory_high_water_mark, preferred_device,
    set_heap_cache_duration, supports_mtl_device, CommandBuffer as MpsCommandBuffer, Predicate,
    PreferredDevice,
};
pub use crate::error::{Error, Result};
pub use crate::filters::{
    HistogramInfo, ImageAdd, ImageBilinearScale, ImageBox, ImageConvolution, ImageGaussianBlur,
    ImageHistogram, ImageLanczosScale, ImageMedian, ImageReduceRowMax, ImageReduceRowMean,
    ImageReduceRowMin, ImageReduceRowSum, ImageScaleAndAdd, ImageSobel, ImageStatisticsMean,
    ImageStatisticsMinAndMax, ImageThresholdBinary, ScaleTransform,
};
pub use crate::image::{
    feature_channel_format, image_edge_mode, image_layout, kernel_options, Image, ImageDescriptor,
    ImageReadWriteParams, ImageRegion,
};
pub use crate::matrix::{
    data_type, data_type_size, Matrix, MatrixDescriptor, MatrixMultiplication,
    MatrixMultiplicationDescriptor, Vector, VectorDescriptor,
};
pub use crate::ndarray::{
    NDArray, NDArrayDescriptor, NDArrayIdentity, NDArrayMatrixMultiplication,
};
pub use crate::neural::{
    cnn_accumulator_precision_option, cnn_convolution_flags, cnn_convolution_weights_layout,
    nn_regularization_type, rnn_bidirectional_combine_mode, rnn_sequence_direction, CnnConvolution,
    CnnConvolutionDescriptor, CnnConvolutionWeightsAndBiasesState, CnnNeuronReluNode,
    CnnPoolingMaxNode, CnnSoftMaxNode, CnnUpsamplingNearestNode, GruDescriptor, LstmDescriptor,
    NNGraph, NNImageNode, NNOptimizer, NNOptimizerAdam, NNOptimizerDescriptor, NNOptimizerRmsProp,
    NNOptimizerStochasticGradientDescent, RnnDescriptor, RnnImageInferenceLayer,
    RnnRecurrentImageState, RnnSingleGateDescriptor,
};
pub use crate::ray::{
    acceleration_structure_status, acceleration_structure_usage, cull_mode, intersection_data_type,
    intersection_type, polygon_type, ray_data_type, winding, PolygonAccelerationStructure,
    RayIntersector, SVGF,
};
pub use crate::state::{
    state_batch_increment_read_count, state_batch_resource_size, state_batch_synchronize,
    state_resource_type, State, StateResourceList, StateTextureInfo,
};
