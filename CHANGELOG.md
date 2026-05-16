# Changelog

## 0.2.1 - 2026-05-16

- Added `State`, `StateResourceList`, `StateTextureInfo`, and state batch helpers for `MPSState`
- Added `NDArrayMatrixMultiplication` plus smoke/example coverage for padded NDArray matrix layouts
- Added executable neural bindings for `CnnConvolution`, `CnnConvolutionWeightsAndBiasesState`, optimizer descriptors/optimizers, and image RNN inference/state APIs
- Added new smoke tests and examples covering NDArray matrix multiply, optimizer/state workflows, convolution, and RNN image inference

## 0.2.0 - 2026-05-16

- Added core device helpers plus `Predicate` and `MpsCommandBuffer`
- Added `NDArrayDescriptor`, `NDArray`, and `NDArrayIdentity`
- Added ray-tracing bindings for `PolygonAccelerationStructure`, `RayIntersector`, and `SVGF`
- Added neural graph bindings for `NNImageNode`, `NNGraph`, ReLU / pooling / softmax / upsampling nodes, `CnnConvolutionDescriptor`, and `RnnSingleGateDescriptor`
- Added Wave-C smoke tests, `COVERAGE.md`, and new NDArray / ray / neural examples

## 0.1.0 - 2026-05-16

- Initial release of `apple-mps`
- Safe wrappers for core image, histogram, reduction, and matrix APIs
- Smoke examples for Gaussian blur and matrix multiplication
