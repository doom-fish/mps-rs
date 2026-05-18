# Changelog

## [0.2.5] - 2026-05-18

- Added one-line rustdoc across the safe wrapper surface, with `///` notes that point back to the corresponding Metal Performance Shaders framework counterparts.
- Hid `ffi` and generated re-exports from public rustdoc so `cargo +nightly rustdoc --lib -- -Z unstable-options --show-coverage` now reports 100.0% documented items.

## [0.2.4] - 2026-05-18

- Widen apple-metal version bound so the 0.x bump dep resolves. No source changes.

## 0.2.3 - 2025-01-16

- Added SAFETY comments to all unsafe blocks across the crate for improved unsafe audit visibility

## 0.2.2 - 2026-05-16

- Closed the MacOSX26.2 umbrella audit to 100% by exposing every remaining public symbol as an executable wrapper, raw-value mirror, or opaque retained handle.
- Added exhaustive `MPSCNN*`, `MPSRNN*`, `MPSNDArray*`, `MPSImage*`, `MPSMatrix*`, `MPSNN*`, `MPSAccelerationStructure*`, `MPSRayIntersector*`, and `MPSState*` family coverage, including deprecated ray-tracing families.
- Added direct image bridge helpers for `MPSGetImageType`, `MPSImageBatchIncrementReadCount`, `MPSImageBatchIterate`, `MPSImageBatchResourceSize`, `MPSImageBatchSynchronize`, and `MPSRectNoClip`.
- Added exhaustive compile/runtime smoke coverage for the generated surface mirrors.

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
