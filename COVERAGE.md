# apple-mps coverage (v0.3.0)

This crate follows the multi-file Swift bridge pattern used in `screencapturekit-rs`:
`@_cdecl` Swift entry points, retained opaque handles returned to Rust, and Rust `Drop`
calling the shared release bridge.

[`COVERAGE_AUDIT.md`](COVERAGE_AUDIT.md) lists the 479 public symbols of the
`MetalPerformanceShaders.h` umbrella in the macOS 26.2 SDK. 97 of them have a usable
wrapper. 353 are opaque retained handles (`as_ptr` and `retained_from_raw` only, no
constructor or methods) and 29 are raw-value newtypes without named constants. Those
382 names exist so that raw pointers can be carried around, but they are not bindings,
and earlier versions of this file counted them as implemented.

## Families

| Family | Status | Notes |
| --- | --- | --- |
| `MPSKernel` | Handle only | `Kernel` is an opaque handle; the kernel wrappers below are separate types. |
| `MPSImage*` | Partial | `Image`, `ImageDescriptor`, validated byte transfers, batch helpers, and the Gaussian blur, box, Sobel, median, convolution, bilinear and Lanczos scale, threshold binary, histogram, statistics, row reduction and add filters. The filters check source and destination compatibility per kernel group before encoding. Canny, morphology, pyramids, integral, conversion, transpose, the remaining thresholds and arithmetic kernels, and `MPSTemporaryImage` are handles only. |
| `MPSMatrix*` | Partial | `Matrix`, `Vector`, their descriptors and `MatrixMultiplication`. Cholesky and LU decomposition, the solvers, FindTopK, softmax, random, copy, fully connected, neuron, sum, batch normalization, matrix-vector multiplication and temporary matrices are handles only. |
| `MPSNDArray*` | Partial | `NDArrayDescriptor`, `NDArray`, `NDArrayIdentity` and `NDArrayMatrixMultiplication`. Gather, strided slice, quantization and the gradient kernels are handles only. |
| `MPSVector*` | Partial | Vector descriptors and buffer-backed vectors; `MPSTemporaryVector` is a handle only. |
| `MPSState*` | Implemented | `State`, `StateResourceList`, `StateTextureInfo` and the state batch helpers. |
| `MPSNNGraph` | Partial | `NNGraph`, `NNImageNode` and the ReLU, max-pooling, softmax and nearest-upsampling nodes. Every other node type is a handle only. |
| `MPSNNOptimizer*` | Partial | SGD, RMSProp and Adam (including AMSGrad) on float32 vectors and matrices; batch-normalization and convolution-state updates are not wrapped. |
| `MPSCNN*` | Partial | `CnnConvolution`, its descriptor and weights state. Pooling, normalization, loss, dropout, fully connected, transposed and binary convolution, arithmetic and the gradient kernels are handles only. |
| `MPSRNN*` | Partial | RNN, GRU and LSTM descriptors and `RnnImageInferenceLayer`; the matrix layers are handles only. |
| `MPSAccelerationStructure`, `MPSRayIntersector`, `MPSPolygonAccelerationStructure` | Partial, deprecated since macOS 14 | Polygon acceleration structures (rebuild, refit) and ray intersection with validated buffers. Instance, triangle and quadrilateral structures, groups and polygon buffers are handles only. |
| `MPSCommandBuffer` | Partial | `MpsCommandBuffer` sets predicates and prefetches the heap, but the encoders take `apple_metal::CommandBuffer`, so kernels never see its predicate. `commitAndContinue` is not exposed because it commits the wrapped buffer behind apple-metal's state tracking. |
| `MPSSVGF` | Properties only | `SVGF` exposes its weights and channel counts; the denoiser has no encode methods, and `MPSTemporalAA` is a handle only. |
| `MPSIntersectionGroup` | Not in the SDK | Absent from the macOS 26.2 headers. |

## Validation

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo +1.82.0 check --lib --all-features
```
