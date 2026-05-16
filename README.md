# mps-rs

Safe Rust bindings for Apple's
[`MetalPerformanceShaders`](https://developer.apple.com/documentation/metalperformanceshaders)
framework on macOS.

The GitHub repository is `mps-rs`; the published crates.io package is
`apple-mps` because the `mps-rs` package name is already taken.

## Install

```bash
cargo add apple-mps apple-metal
```

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
blur.encode_image(&command_buffer, &src, &dst);
```

## v0.2 surface

- Core helpers:
  - `supports_mtl_device`, `preferred_device`, `hint_temporary_memory_high_water_mark`, `set_heap_cache_duration`
  - `Predicate` and `MpsCommandBuffer`
- Images:
  - `ImageDescriptor` + `Image` for lazily allocated MPS images or texture-backed images
  - Float32 image read/write helpers plus raw byte transfer with `MPSDataLayout`
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
  - `NDArrayDescriptor`, `NDArray`, and `NDArrayIdentity`
- Ray tracing / denoising:
  - `PolygonAccelerationStructure`, `RayIntersector`, and `SVGF`
- Neural graph / descriptors:
  - `NNImageNode`, `NNGraph`
  - `CnnNeuronReluNode`, `CnnPoolingMaxNode`, `CnnSoftMaxNode`, `CnnUpsamplingNearestNode`
  - `CnnConvolutionDescriptor`, `RnnSingleGateDescriptor`
- Shared constants for `MPSKernelOptions`, `MPSImageEdgeMode`, `MPSImageFeatureChannelFormat`, `MPSDataType`, `MPSDataLayout`, and new ray/RNN enums

See [`COVERAGE.md`](COVERAGE.md) for the Wave-C audit and the implemented/partial matrix.

## Validation

```bash
cargo test
cargo run --example 01_blur_image
cargo run --example 02_matrix_multiply
cargo run --example 03_ndarray_identity
cargo run --example 04_ray_intersection
cargo run --example 05_nn_graph_relu
```
