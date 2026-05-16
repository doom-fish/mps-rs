# apple-mps coverage (v0.2.0)

This crate follows the multi-file Swift bridge pattern used in `screencapturekit-rs`:
`@_cdecl` Swift entry points, retained opaque handles returned to Rust, and Rust `Drop`
calling the shared release bridge.

## Requested Wave-C audit

| Family | Status | Notes |
| --- | --- | --- |
| `MPSKernel` | Partial | Kernel-derived wrappers are exposed across image, matrix, ray, and neural APIs. Direct base-class property bridging is still limited to shared constants such as `kernel_options`. |
| `MPSImage*` | Implemented | `Image`, `ImageDescriptor`, read/write helpers, and a broad set of unary/binary image kernels are exposed. |
| `MPSMatrix*` | Implemented | `MatrixDescriptor`, `VectorDescriptor`, `Matrix`, `Vector`, `MatrixMultiplication`, and `MatrixMultiplicationDescriptor` are exposed. |
| `MPSNDArray*` | Partial | `NDArrayDescriptor`, `NDArray`, and `NDArrayIdentity` are exposed. Matrix-multiply, gather, slice, and quantization kernels are not yet bound. |
| `MPSVector*` | Implemented | Vector descriptors and buffer-backed vector wrappers are exposed in `matrix.rs`. |
| `MPSNNGraph` | Partial | `NNImageNode`, `NNGraph`, and image-only graph encoding are exposed. Current graph binding targets image inputs and result images. |
| `MPSCNN*` | Partial | Existing image kernels remain available, and v0.2.0 adds `CnnNeuronReluNode`, `CnnPoolingMaxNode`, `CnnSoftMaxNode`, `CnnUpsamplingNearestNode`, and `CnnConvolutionDescriptor`. Direct convolution/filter execution APIs that require `MPSCNNConvolutionDataSource` are not yet wrapped. |
| `MPSRNN*` | Partial | `RnnSingleGateDescriptor` is exposed. Image/matrix inference and training layers are not yet wrapped. |
| `MPSAccelerationStructure` | Partial | Common acceleration-structure functionality (`usage`, `status`, `rebuild`, `encode_refit`) is exposed through `PolygonAccelerationStructure`. A standalone base wrapper and group APIs are not yet exposed. |
| `MPSRayIntersector` | Implemented | `RayIntersector` exposes core configuration, recommended batch sizing, and `encode_intersection`. |
| `MPSPolygonAccelerationStructure` | Implemented | Polygon type, buffer wiring, counts, rebuild, and refit support are exposed. |
| `MPSCommandBuffer` | Implemented | `MpsCommandBuffer` and `Predicate` wrappers are exposed in `core.rs`. |
| `MPSSVGF` | Partial | Construction plus core tuning properties (`depth_weight`, `normal_weight`, `luminance_weight`, `channel_count`, `channel_count2`) are exposed. Reprojection and bilateral-encode entry points are not yet wrapped. |
| `MPSIntersectionGroup` | Unavailable in current SDK | Not present in the local macOS 26.2 SDK headers under `MPSRayIntersector.framework/Headers`. |

## Validation

The v0.2.0 sweep is validated with:

```bash
cargo clippy --all-targets -- -D warnings
cargo test
cargo run --example 01_blur_image
cargo run --example 02_matrix_multiply
cargo run --example 03_ndarray_identity
cargo run --example 04_ray_intersection
cargo run --example 05_nn_graph_relu
```
