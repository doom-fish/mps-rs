# mps-rs

Safe Rust bindings for Apple's
[`MetalPerformanceShaders`](https://developer.apple.com/documentation/metalperformanceshaders)
framework on macOS.

The GitHub repository is `mps-rs`; the published crates.io package is
`apple-mps` because the `mps-rs` package name is already taken.

## Install

```toml
[dependencies]
apple-mps = "0.3"
apple-metal = "0.10"
```

Requires macOS 11 or later and Rust 1.82. Buffer-backed `NDArray`s and
`NDArrayIdentity` need macOS 15 and report `Error::Unsupported` or `None` on
older systems. The ray-tracing types wrap APIs Apple deprecated in macOS 14.

## Quick start

```rust,no_run
use apple_metal::MetalDevice;
use apple_mps::{feature_channel_format, Image, ImageDescriptor, ImageGaussianBlur};

let device = MetalDevice::system_default().expect("no Metal device");
let queue = device.new_command_queue().expect("queue");
let descriptor = ImageDescriptor::new(256, 256, 1, feature_channel_format::FLOAT32);
let src = Image::new(&device, descriptor).expect("source image");
let dst = Image::new(&device, descriptor).expect("destination image");
let blur = ImageGaussianBlur::new(&device, 2.0).expect("gaussian blur");
let command_buffer = queue.new_command_buffer().expect("command buffer");
blur.encode_image(&command_buffer, &src, &dst).expect("encode blur");
```

## Surface

- Only part of the umbrella is usable: most kernel classes are opaque handles without
  constructors or encode methods. [`COVERAGE.md`](COVERAGE.md) lists what is wrapped.
- Core helpers:
  - `supports_mtl_device`, `preferred_device`, `hint_temporary_memory_high_water_mark`, `set_heap_cache_duration`
  - `Predicate` and `MpsCommandBuffer` (predicates and heap prefetch only; the encoders
    take `apple_metal::CommandBuffer`, so kernels never see an `MpsCommandBuffer` predicate)
- Images:
  - `ImageDescriptor` + `Image` for lazily allocated MPS images or texture-backed images
  - Float32 image read/write helpers plus raw byte transfers with `MPSDataLayout`, checked against the image's channel format, region and feature-channel window
  - Unary image filters:
    - `ImageGaussianBlur`
    - `ImageBox`
    - `ImageSobel`
    - `ImageMedian`
    - `ImageConvolution`
    - `ImageBilinearScale`
    - `ImageLanczosScale`
    - `ImageThresholdBinary`
    - `ImageStatisticsMinAndMax`
    - `ImageStatisticsMean`
    - `ImageReduceRowMin`, `ImageReduceRowMax`, `ImageReduceRowMean`, `ImageReduceRowSum`
  - `ImageHistogram`, `ImageAdd`, and `ImageScaleAndAdd`
- Matrix/vector:
  - `MatrixDescriptor`, `VectorDescriptor`, `MatrixMultiplicationDescriptor`, `Matrix`, `Vector`, and `MatrixMultiplication`
- `NDArray`:
  - `NDArrayDescriptor`, `NDArray`, `NDArrayIdentity`, and `NDArrayMatrixMultiplication`
- State:
  - `State`, `StateResourceList`, `StateTextureInfo`
  - `state_batch_increment_read_count`, `state_batch_resource_size`, `state_batch_synchronize`
- Ray tracing / denoising:
  - `PolygonAccelerationStructure`, `RayIntersector`, and `SVGF`
- Neural / optimizer / RNN:
  - `NNImageNode`, `NNGraph`
  - `CnnNeuronReluNode`, `CnnPoolingMaxNode`, `CnnSoftMaxNode`, `CnnUpsamplingNearestNode`
  - `CnnConvolutionDescriptor`, `CnnConvolution`, `CnnConvolutionWeightsAndBiasesState`
  - `NNOptimizerDescriptor`, `NNOptimizer`, `NNOptimizerStochasticGradientDescent`, `NNOptimizerRmsProp`, `NNOptimizerAdam`
  - `RnnDescriptor`, `RnnSingleGateDescriptor`, `GruDescriptor`, `LstmDescriptor`, `RnnImageInferenceLayer`, `RnnRecurrentImageState`
- Shared constants for `MPSKernelOptions`, `MPSImageEdgeMode`, `MPSImageFeatureChannelFormat`, `MPSDataType`, `MPSDataLayout`, plus convolution / optimizer / RNN / state enums

See [`COVERAGE.md`](COVERAGE.md) for the family matrix and what the coverage audit measures.

## Safety notes

- Constructors and encode methods check buffer lengths, strides, offsets, shapes and data
  types before calling MPS and return `Err` (or `None`) instead of letting MPS read or write
  out of bounds or abort the process.
- Image filters check their textures against each kernel's rules before encoding (pixel
  format class, color model or channel count, texture type, feature channels, aliasing,
  shader usage, device and, for min/max statistics, destination width) and return `Err`.
- Every encode checks that the command buffer is still recording. MPS still aborts if
  an apple-metal encoder is open on the same command buffer, or if another thread
  commits it during an encode: apple-metal does not expose its encoder state to other
  crates, so end your encoders before encoding MPS work.
- MPS kernels and descriptors are `Send` but not `Sync`: MPS allows a kernel to be used by
  one thread at a time. Data objects such as `Matrix`, `Vector`, `NDArray` and `Image`
  remain `Sync`.
- `PolygonAccelerationStructure::set_index_buffer` is `unsafe`: every index read for the
  configured polygons must be smaller than the number of vertices in the vertex buffer
  whenever the structure is rebuilt, refit or used, because MPS reads vertices through
  the indices without checking them.
- As Apple documents, GPU writes to geometry, ray and image data must have completed
  before `rebuild` or a CPU image transfer reads it.

## Validation

```bash
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all-features
cargo run --example 01_blur_image
cargo run --example 02_matrix_multiply
cargo run --example 03_ndarray_identity
cargo run --example 04_ray_intersection
cargo run --example 05_nn_graph_relu
cargo run --example 06_ndarray_matrix_multiplication
cargo run --example 07_optimizer_and_state
cargo run --example 08_cnn_convolution
cargo run --example 09_rnn_image_inference
```
