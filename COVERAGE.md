# apple-mps coverage (v0.2.2)

This crate follows the multi-file Swift bridge pattern used in `screencapturekit-rs`:
`@_cdecl` Swift entry points, retained opaque handles returned to Rust, and Rust `Drop`
calling the shared release bridge.

## Requested Wave-C audit

| Family | Status | Notes |
| --- | --- | --- |
| `MPSKernel` | Implemented | `Kernel` now complements the existing kernel-derived image, matrix, ray, and neural wrappers. |
| `MPSImage*` | Implemented | `Image`, `ImageDescriptor`, read/write helpers, image batch helpers, `get_image_type`, and exhaustive image-family handle/raw-value mirrors are exposed. |
| `MPSMatrix*` | Implemented | Runtime matrix/vector wrappers are joined by exhaustive matrix-family handle/raw-value mirrors, including temporary and decomposition/random families. |
| `MPSNDArray*` | Implemented | `NDArrayDescriptor`, `NDArray`, `NDArrayIdentity`, `NDArrayMatrixMultiplication`, and the remaining gather/slice/quantization families are exposed as runtime wrappers or opaque handles. |
| `MPSVector*` | Implemented | Vector descriptors and buffer-backed vector wrappers remain available, alongside temporary-vector family coverage. |
| `MPSState*` | Implemented | `State`, `StateResourceList`, `StateTextureInfo`, and state batch helpers now sit inside a fully covered state-family audit surface. |
| `MPSNNGraph` | Implemented | `NNImageNode`, `NNGraph`, and the remaining `MPSNN*` graph/node/state/protocol surface are now exposed. |
| `MPSNNOptimizer*` | Implemented | Optimizer descriptors/optimizers remain executable, with the remaining optimizer-adjacent graph/trainable symbols now mirrored exhaustively. |
| `MPSCNN*` | Implemented | Executable convolution wrappers are complemented by exhaustive opaque-handle/raw-value coverage for the rest of the CNN family, including data-source and gradient types. |
| `MPSRNN*` | Implemented | Existing descriptor/image-layer wrappers are extended by matrix-layer/state/raw-value coverage across the full RNN family. |
| `MPSAccelerationStructure` | Implemented | Existing polygon acceleration functionality is complemented by exhaustive base/group/instance/triangle/quadrilateral accelerator mirrors, including deprecated variants. |
| `MPSRayIntersector` | Implemented | `RayIntersector` now sits alongside mask/test/temporal/raw-value mirrors for the full ray-intersector family. |
| `MPSPolygonAccelerationStructure` | Implemented | Polygon type, buffer wiring, counts, rebuild, and refit support remain available and now participate in the exhaustive audit surface. |
| `MPSCommandBuffer` | Implemented | `MpsCommandBuffer` and `Predicate` wrappers remain exposed in `core.rs`. |
| `MPSSVGF` | Implemented | `SVGF` plus denoiser/allocator/temporal family mirrors are now covered. |
| `MPSIntersectionGroup` | Unavailable in current SDK | Not present in the local macOS 26.2 SDK headers under `MPSRayIntersector.framework/Headers`. |

## Validation

The v0.2.2 exhaustive sweep is validated with:

```bash
cargo clippy --all-targets -- -D warnings
cargo test
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
