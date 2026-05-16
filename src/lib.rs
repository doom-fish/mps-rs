#![doc = include_str!("../README.md")]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![allow(clippy::missing_const_for_fn)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]

pub mod error;
pub mod ffi;
pub mod filters;
pub mod image;
pub mod matrix;

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
