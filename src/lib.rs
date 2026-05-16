#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod core;
pub mod error;
pub mod ffi;
pub mod filters;
pub mod image;
pub mod matrix;
pub mod ndarray;
pub mod neural;
pub mod ray;

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
pub use crate::ndarray::{NDArray, NDArrayDescriptor, NDArrayIdentity};
pub use crate::neural::{
    rnn_sequence_direction, CnnConvolutionDescriptor, CnnNeuronReluNode, CnnPoolingMaxNode,
    CnnSoftMaxNode, CnnUpsamplingNearestNode, NNGraph, NNImageNode, RnnSingleGateDescriptor,
};
pub use crate::ray::{
    acceleration_structure_status, acceleration_structure_usage, cull_mode, intersection_data_type,
    intersection_type, polygon_type, ray_data_type, winding, PolygonAccelerationStructure,
    RayIntersector, SVGF,
};
