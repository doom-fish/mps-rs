use crate::ffi;
use core::ffi::c_void;
use core::ptr;

macro_rules! opaque_generated_handle {
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

            /// # Safety
            ///
            /// `ptr` must be a valid Objective-C object of the matching MPS type or protocol.
            #[must_use]
            pub unsafe fn retained_from_raw(ptr: *mut c_void) -> Option<Self> {
                let retained = unsafe { ffi::mps_object_retain(ptr) };
                if retained.is_null() {
                    None
                } else {
                    Some(Self { ptr: retained })
                }
            }
        }
    };
}

macro_rules! raw_value_type {
    ($name:ident, $raw:ty) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
        #[repr(transparent)]
        pub struct $name(pub $raw);

        impl $name {
            #[must_use]
            pub const fn from_raw(raw: $raw) -> Self {
                Self(raw)
            }

            #[must_use]
            pub const fn as_raw(self) -> $raw {
                self.0
            }
        }

        impl From<$raw> for $name {
            fn from(value: $raw) -> Self {
                Self(value)
            }
        }

        impl From<$name> for $raw {
            fn from(value: $name) -> Self {
                value.0
            }
        }
    };
}

raw_value_type!(CnnBatchNormalizationFlags, usize);
raw_value_type!(CnnConvolutionGradientOption, usize);
raw_value_type!(NNComparisonType, usize);
raw_value_type!(NNPaddingMethod, usize);
raw_value_type!(NNTrainingStyle, usize);
raw_value_type!(CnnBinaryConvolutionFlags, usize);
raw_value_type!(CnnBinaryConvolutionType, usize);
raw_value_type!(CnnLossType, usize);
raw_value_type!(CnnNeuronType, usize);
raw_value_type!(CnnReductionType, usize);
raw_value_type!(CnnWeightsQuantizationType, usize);
raw_value_type!(RnnMatrixId, usize);
opaque_generated_handle!(CnnAdd);
opaque_generated_handle!(CnnAddGradient);
opaque_generated_handle!(CnnArithmetic);
opaque_generated_handle!(CnnArithmeticGradient);
opaque_generated_handle!(CnnArithmeticGradientState);
opaque_generated_handle!(CnnBatchNormalization);
opaque_generated_handle!(CnnBatchNormalizationGradient);
opaque_generated_handle!(CnnBatchNormalizationGradientNode);
opaque_generated_handle!(CnnBatchNormalizationState);
opaque_generated_handle!(CnnBatchNormalizationStatistics);
opaque_generated_handle!(CnnBatchNormalizationStatisticsGradient);
opaque_generated_handle!(CnnBinaryConvolution);
opaque_generated_handle!(CnnBinaryConvolutionNode);
opaque_generated_handle!(CnnBinaryFullyConnected);
opaque_generated_handle!(CnnBinaryFullyConnectedNode);
opaque_generated_handle!(CnnBinaryKernel);
opaque_generated_handle!(CnnConvolutionGradient);
opaque_generated_handle!(CnnConvolutionGradientNode);
opaque_generated_handle!(CnnConvolutionGradientStateNode);
opaque_generated_handle!(CnnConvolutionTranspose);
opaque_generated_handle!(CnnConvolutionTransposeGradient);
opaque_generated_handle!(CnnConvolutionTransposeGradientNode);
opaque_generated_handle!(CnnConvolutionTransposeGradientState);
opaque_generated_handle!(CnnConvolutionTransposeGradientStateNode);
opaque_generated_handle!(CnnConvolutionTransposeNode);
opaque_generated_handle!(CnnCrossChannelNormalization);
opaque_generated_handle!(CnnCrossChannelNormalizationGradient);
opaque_generated_handle!(CnnCrossChannelNormalizationGradientNode);
opaque_generated_handle!(CnnCrossChannelNormalizationNode);
opaque_generated_handle!(CnnDepthWiseConvolutionDescriptor);
opaque_generated_handle!(CnnDilatedPoolingMax);
opaque_generated_handle!(CnnDilatedPoolingMaxGradient);
opaque_generated_handle!(CnnDilatedPoolingMaxGradientNode);
opaque_generated_handle!(CnnDilatedPoolingMaxNode);
opaque_generated_handle!(CnnDivide);
opaque_generated_handle!(CnnDropout);
opaque_generated_handle!(CnnDropoutGradient);
opaque_generated_handle!(CnnDropoutGradientNode);
opaque_generated_handle!(CnnDropoutGradientState);
opaque_generated_handle!(CnnDropoutNode);
opaque_generated_handle!(CnnFullyConnected);
opaque_generated_handle!(CnnFullyConnectedGradient);
opaque_generated_handle!(CnnFullyConnectedGradientNode);
opaque_generated_handle!(CnnFullyConnectedNode);
opaque_generated_handle!(CnnGradientKernel);
opaque_generated_handle!(CnnGroupNormalization);
opaque_generated_handle!(CnnGroupNormalizationGradient);
opaque_generated_handle!(CnnGroupNormalizationGradientNode);
opaque_generated_handle!(CnnGroupNormalizationGradientState);
opaque_generated_handle!(CnnInstanceNormalization);
opaque_generated_handle!(CnnInstanceNormalizationGradient);
opaque_generated_handle!(CnnInstanceNormalizationGradientNode);
opaque_generated_handle!(CnnInstanceNormalizationGradientState);
opaque_generated_handle!(CnnKernel);
opaque_generated_handle!(CnnLocalContrastNormalization);
opaque_generated_handle!(CnnLocalContrastNormalizationGradient);
opaque_generated_handle!(CnnLocalContrastNormalizationGradientNode);
opaque_generated_handle!(CnnLocalContrastNormalizationNode);
opaque_generated_handle!(CnnLogSoftMax);
opaque_generated_handle!(CnnLogSoftMaxGradient);
opaque_generated_handle!(CnnLogSoftMaxGradientNode);
opaque_generated_handle!(CnnLogSoftMaxNode);
opaque_generated_handle!(CnnLoss);
opaque_generated_handle!(CnnLossDataDescriptor);
opaque_generated_handle!(CnnLossDescriptor);
opaque_generated_handle!(CnnLossLabels);
opaque_generated_handle!(CnnLossNode);
opaque_generated_handle!(CnnMultiaryKernel);
opaque_generated_handle!(CnnMultiply);
opaque_generated_handle!(CnnMultiplyGradient);
opaque_generated_handle!(CnnNeuron);
opaque_generated_handle!(CnnNeuronAbsolute);
opaque_generated_handle!(CnnNeuronAbsoluteNode);
opaque_generated_handle!(CnnNeuronELU);
opaque_generated_handle!(CnnNeuronELUNode);
opaque_generated_handle!(CnnNeuronExponential);
opaque_generated_handle!(CnnNeuronExponentialNode);
opaque_generated_handle!(CnnNeuronGeLUNode);
opaque_generated_handle!(CnnNeuronGradient);
opaque_generated_handle!(CnnNeuronGradientNode);
opaque_generated_handle!(CnnNeuronHardSigmoid);
opaque_generated_handle!(CnnNeuronHardSigmoidNode);
opaque_generated_handle!(CnnNeuronLinear);
opaque_generated_handle!(CnnNeuronLinearNode);
opaque_generated_handle!(CnnNeuronLogarithm);
opaque_generated_handle!(CnnNeuronLogarithmNode);
opaque_generated_handle!(CnnNeuronNode);
opaque_generated_handle!(CnnNeuronPReLU);
opaque_generated_handle!(CnnNeuronPReLUNode);
opaque_generated_handle!(CnnNeuronPower);
opaque_generated_handle!(CnnNeuronPowerNode);
opaque_generated_handle!(CnnNeuronReLU);
opaque_generated_handle!(CnnNeuronReLUN);
opaque_generated_handle!(CnnNeuronReLUNNode);
opaque_generated_handle!(CnnNeuronSigmoid);
opaque_generated_handle!(CnnNeuronSigmoidNode);
opaque_generated_handle!(CnnNeuronSoftPlus);
opaque_generated_handle!(CnnNeuronSoftPlusNode);
opaque_generated_handle!(CnnNeuronSoftSign);
opaque_generated_handle!(CnnNeuronSoftSignNode);
opaque_generated_handle!(CnnNeuronTanH);
opaque_generated_handle!(CnnNeuronTanHNode);
opaque_generated_handle!(CnnNormalizationGammaAndBetaState);
opaque_generated_handle!(CnnNormalizationMeanAndVarianceState);
opaque_generated_handle!(CnnNormalizationNode);
opaque_generated_handle!(CnnPooling);
opaque_generated_handle!(CnnPoolingAverage);
opaque_generated_handle!(CnnPoolingAverageGradient);
opaque_generated_handle!(CnnPoolingAverageGradientNode);
opaque_generated_handle!(CnnPoolingAverageNode);
opaque_generated_handle!(CnnPoolingGradient);
opaque_generated_handle!(CnnPoolingGradientNode);
opaque_generated_handle!(CnnPoolingL2Norm);
opaque_generated_handle!(CnnPoolingL2NormGradient);
opaque_generated_handle!(CnnPoolingL2NormGradientNode);
opaque_generated_handle!(CnnPoolingL2NormNode);
opaque_generated_handle!(CnnPoolingMax);
opaque_generated_handle!(CnnPoolingMaxGradient);
opaque_generated_handle!(CnnPoolingMaxGradientNode);
opaque_generated_handle!(CnnPoolingNode);
opaque_generated_handle!(CnnSoftMax);
opaque_generated_handle!(CnnSoftMaxGradient);
opaque_generated_handle!(CnnSoftMaxGradientNode);
opaque_generated_handle!(CnnSpatialNormalization);
opaque_generated_handle!(CnnSpatialNormalizationGradient);
opaque_generated_handle!(CnnSpatialNormalizationGradientNode);
opaque_generated_handle!(CnnSpatialNormalizationNode);
opaque_generated_handle!(CnnSubPixelConvolutionDescriptor);
opaque_generated_handle!(CnnSubtract);
opaque_generated_handle!(CnnSubtractGradient);
opaque_generated_handle!(CnnUpsampling);
opaque_generated_handle!(CnnUpsamplingBilinear);
opaque_generated_handle!(CnnUpsamplingBilinearGradient);
opaque_generated_handle!(CnnUpsamplingBilinearGradientNode);
opaque_generated_handle!(CnnUpsamplingBilinearNode);
opaque_generated_handle!(CnnUpsamplingGradient);
opaque_generated_handle!(CnnUpsamplingNearest);
opaque_generated_handle!(CnnUpsamplingNearestGradient);
opaque_generated_handle!(CnnUpsamplingNearestGradientNode);
opaque_generated_handle!(CnnYOLOLoss);
opaque_generated_handle!(CnnYOLOLossDescriptor);
opaque_generated_handle!(CnnYOLOLossNode);
opaque_generated_handle!(NNAdditionGradientNode);
opaque_generated_handle!(NNAdditionNode);
opaque_generated_handle!(NNArithmeticGradientNode);
opaque_generated_handle!(NNArithmeticGradientStateNode);
opaque_generated_handle!(NNBilinearScaleNode);
opaque_generated_handle!(NNBinaryArithmeticNode);
opaque_generated_handle!(NNBinaryGradientState);
opaque_generated_handle!(NNBinaryGradientStateNode);
opaque_generated_handle!(NNCompare);
opaque_generated_handle!(NNComparisonNode);
opaque_generated_handle!(NNConcatenationGradientNode);
opaque_generated_handle!(NNConcatenationNode);
opaque_generated_handle!(NNCropAndResizeBilinear);
opaque_generated_handle!(NNDefaultPadding);
opaque_generated_handle!(NNDivisionNode);
opaque_generated_handle!(NNFilterNode);
opaque_generated_handle!(NNForwardLoss);
opaque_generated_handle!(NNForwardLossNode);
opaque_generated_handle!(NNGradientFilterNode);
opaque_generated_handle!(NNGradientState);
opaque_generated_handle!(NNGradientStateNode);
opaque_generated_handle!(NNGramMatrixCalculation);
opaque_generated_handle!(NNGramMatrixCalculationGradient);
opaque_generated_handle!(NNGramMatrixCalculationGradientNode);
opaque_generated_handle!(NNGramMatrixCalculationNode);
opaque_generated_handle!(NNGridSample);
opaque_generated_handle!(NNInitialGradient);
opaque_generated_handle!(NNInitialGradientNode);
opaque_generated_handle!(NNLabelsNode);
opaque_generated_handle!(NNLanczosScaleNode);
opaque_generated_handle!(NNLocalCorrelation);
opaque_generated_handle!(NNLossGradient);
opaque_generated_handle!(NNLossGradientNode);
opaque_generated_handle!(NNMultiaryGradientState);
opaque_generated_handle!(NNMultiaryGradientStateNode);
opaque_generated_handle!(NNMultiplicationGradientNode);
opaque_generated_handle!(NNMultiplicationNode);
opaque_generated_handle!(NNNeuronDescriptor);
opaque_generated_handle!(NNPad);
opaque_generated_handle!(NNPadGradient);
opaque_generated_handle!(NNPadGradientNode);
opaque_generated_handle!(NNPadNode);
opaque_generated_handle!(NNReduceBinary);
opaque_generated_handle!(NNReduceColumnMax);
opaque_generated_handle!(NNReduceColumnMean);
opaque_generated_handle!(NNReduceColumnMin);
opaque_generated_handle!(NNReduceColumnSum);
opaque_generated_handle!(NNReduceFeatureChannelsAndWeightsMean);
opaque_generated_handle!(NNReduceFeatureChannelsAndWeightsSum);
opaque_generated_handle!(NNReduceFeatureChannelsArgumentMax);
opaque_generated_handle!(NNReduceFeatureChannelsArgumentMin);
opaque_generated_handle!(NNReduceFeatureChannelsMax);
opaque_generated_handle!(NNReduceFeatureChannelsMean);
opaque_generated_handle!(NNReduceFeatureChannelsMin);
opaque_generated_handle!(NNReduceFeatureChannelsSum);
opaque_generated_handle!(NNReduceRowMax);
opaque_generated_handle!(NNReduceRowMean);
opaque_generated_handle!(NNReduceRowMin);
opaque_generated_handle!(NNReduceRowSum);
opaque_generated_handle!(NNReduceUnary);
opaque_generated_handle!(NNReductionColumnMaxNode);
opaque_generated_handle!(NNReductionColumnMeanNode);
opaque_generated_handle!(NNReductionColumnMinNode);
opaque_generated_handle!(NNReductionColumnSumNode);
opaque_generated_handle!(NNReductionFeatureChannelsArgumentMaxNode);
opaque_generated_handle!(NNReductionFeatureChannelsArgumentMinNode);
opaque_generated_handle!(NNReductionFeatureChannelsMaxNode);
opaque_generated_handle!(NNReductionFeatureChannelsMeanNode);
opaque_generated_handle!(NNReductionFeatureChannelsMinNode);
opaque_generated_handle!(NNReductionFeatureChannelsSumNode);
opaque_generated_handle!(NNReductionRowMaxNode);
opaque_generated_handle!(NNReductionRowMeanNode);
opaque_generated_handle!(NNReductionRowMinNode);
opaque_generated_handle!(NNReductionRowSumNode);
opaque_generated_handle!(NNReductionSpatialMeanGradientNode);
opaque_generated_handle!(NNReductionSpatialMeanNode);
opaque_generated_handle!(NNReshape);
opaque_generated_handle!(NNReshapeGradient);
opaque_generated_handle!(NNReshapeGradientNode);
opaque_generated_handle!(NNReshapeNode);
opaque_generated_handle!(NNResizeBilinear);
opaque_generated_handle!(NNScaleNode);
opaque_generated_handle!(NNSlice);
opaque_generated_handle!(NNStateNode);
opaque_generated_handle!(NNSubtractionGradientNode);
opaque_generated_handle!(NNSubtractionNode);
opaque_generated_handle!(NNUnaryReductionNode);
opaque_generated_handle!(RnnMatrixInferenceLayer);
opaque_generated_handle!(RnnMatrixTrainingLayer);
opaque_generated_handle!(RnnMatrixTrainingState);
opaque_generated_handle!(RnnRecurrentMatrixState);
opaque_generated_handle!(CnnBatchNormalizationDataSource);
opaque_generated_handle!(CnnConvolutionDataSource);
opaque_generated_handle!(CnnGroupNormalizationDataSource);
opaque_generated_handle!(CnnInstanceNormalizationDataSource);
opaque_generated_handle!(Handle);
opaque_generated_handle!(NNGramMatrixCallback);
opaque_generated_handle!(NNLossCallback);
opaque_generated_handle!(NNPadding);
opaque_generated_handle!(NNTrainableNode);
