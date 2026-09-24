# Changelog

All notable changes to `apple-mps` are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - Unreleased

### Security

- `Image::read_bytes` and `write_bytes` checked the slice only against
  `bytes_per_row * height * depth` and never required `bytes_per_row` to hold a
  row, so a 4-byte slice with a 1-byte stride let MPS write (or read) a whole
  4x4 float region past the slice, and the bridge's `Bool` was ignored.
  Transfers are now checked against the image's feature-channel format, the
  region, the image index and the MPS feature-channel window rules with checked
  arithmetic, the bridge checks again, and a refusal is an error instead of
  `Ok(())`.
- `Matrix`, `Vector` and `NDArray::new_with_buffer` never compared the described
  storage with `buffer.length()`, so a 4-byte buffer could back a 1024x1024
  matrix and kernels read and wrote outside it on the GPU. They now check the
  exact extent, that strides are element multiples large enough for a row, a
  matrix or a vector, and for NDArrays the offset and MPS's 16-byte-padded row
  layout.
- Ray-tracing vertex, index, ray and intersection buffers, offsets, strides and
  counts were unchecked, so builds read past the vertex buffer and intersections
  were written past their buffer. They are now checked against the SDK struct
  sizes and alignments, and intersections and refits need a current rebuild.
- `ImageHistogram` encodes wrote past a short histogram buffer on the GPU
  without an error; the buffer and its 32-byte aligned offset are now checked.

### Fixed

- MPS assertions and Swift traps that aborted the process from safe code:
  convolution groups of 0, larger than the input channel count, not dividing
  both channel counts or leaving fewer than a multiple of 4 channels per group
  (MPS checks in `setGroups:` itself); zero kernel sizes, strides or channel
  counts; M/N/K mismatches and unsupported data-type combinations in
  `MatrixMultiplication::encode`; wrong source counts and interior, batch,
  addend or destination shapes in `NDArrayMatrixMultiplication`; NDArray
  descriptors with more than 16 dimensions or 2^31 elements, out-of-range
  transposes, `NDArray::length_of_dimension` past the rank, and identity
  reshapes that change the volume or miss the destination shape; image
  descriptors with zero or oversized extents, more than 2048 slices, no channel
  format, memoryless storage or unknown usage bits; texture-backed images whose
  channel count doesn't fit the texture; reading an image that was never
  written, or encoding a convolution on one; `NNGraph::encode` with too few
  images; optimizer operands that differ in shape or are not float32; refitting
  a structure built without `REFIT` usage; `MPSState` indices past
  `resource_count`; predicates outside their buffer; histogram bin counts that
  are not a power of two; and `Int`/`Int32` conversions of NDArray sizes and
  LSTM neuron types.
- The bridge's enum guards never rejected anything because Swift's imported
  `init?(rawValue:)` accepts any raw value; the ray-tracing setters and image
  channel formats now range-check their raw values.
- Graph image nodes now carry an `MPSHandle`, so `NNGraph::source_image_count`
  reports the images the graph needs instead of 0.
- Image filters passed incompatible textures to MPS, which asserts and aborts
  the process. Before encoding, the unary filters, `ImageAdd` and
  `ImageHistogram` now check, per kernel group: that the command buffer is
  still recording; normalized or floating-point formats (integer, depth,
  compressed and packed 16-bit destination formats are refused); the color
  model or channel count each kernel requires (Sobel may reduce to one
  channel, thresholding converts anything, the statistics kernels need equal
  channel counts, median and statistics refuse A8); matching 2D or 2D-array
  texture types and array lengths; at most 4 feature channels for unary
  kernels; a destination that does not alias a source (including views of
  it); sources readable and destinations writable by shaders; a clipped
  destination at least 2 pixels wide for `ImageStatisticsMinAndMax`; the
  kernel's device; and a 2D histogram source.
- `ImageBox::new` and `ImageConvolution::new` aborted on even kernel sizes and
  `ImageMedian::new` on even diameters or diameters outside
  `MPSImageMedian`'s 3–127 range; they now return `None`.
- `ImageHistogram::histogram_size_for_source_format` aborted for formats MPS
  cannot histogram (A8, depth, compressed); it now returns an error.
- `ImageHistogram` accepted integer source textures, which its kernels read
  through a float texture parameter: the results are undefined and Metal's
  validation layer aborts. `encode_image`, `encode_texture` and
  `histogram_size_for_source_format` now refuse integer formats.
- Encoding into a committed command buffer aborted, and so did encoding while an
  apple-metal encoder was open on it ("A command encoder is already encoding to
  this command buffer") or while another thread committed it
  (`_status < MTLCommandBufferStatusCommitted` in `setCurrentCommandEncoder:`,
  reproduced by a 4096-iteration commit race). Every encode path, the temporary
  `State` constructors, the synchronize helpers,
  `hint_temporary_memory_high_water_mark`, `set_heap_cache_duration` and
  `MpsCommandBuffer::prefetch_heap_for_workload_size` now run inside apple-metal's
  `CommandBuffer::encode_foreign`, which refuses a buffer that no longer records
  or has an open encoder and holds off commits, enqueues and new encoders until
  MPS returns.
- Dropping a temporary `State` whose read count was above zero made the Metal
  validation layer abort the process ("was released before its readCount was
  zero"); `tests/v021_smoke.rs` hit this under `MTL_DEBUG_LAYER=1`. Dropping a
  `State` now sets a temporary state's read count to zero first, which returns
  its storage to MPS as Apple documents.
- MPS aborted, even without the validation layer, when `State::set_read_count`
  was called on a persistent state or raised a count that had reached zero, when
  `state_batch_increment_read_count` would take a count below zero, and when
  `synchronize_on_command_buffer` or `state_batch_synchronize` met a temporary
  state, whose storage is GPU-private. These now return `Error::InvalidArgument`.
- `NDArrayMatrixMultiplication::encode` returned an `MPSTemporaryNDArray` from
  MPS's default destination allocator. Reading the result twice tripped the
  validation layer, and its contents became undefined once its read count reached
  zero. The kernel now allocates regular `MPSNDArray` results.
- Temporary recurrent outputs of `RnnImageInferenceLayer` hold temporary images
  with their own read counts, and the validation layer aborted on every use, even
  after the state's count was zeroed. `set_recurrent_output_is_temporary(true)`
  now returns `Error::Unsupported`.
- `NNGraph::encode` left every intermediate image exported with
  `NNImageNode::set_export_from_graph(true)` as an unreachable `MPSTemporaryImage`
  with read count 1, so the validation layer aborted when MPS released it. The
  encode now collects the exported images and sets their read counts to zero.
- COVERAGE.md and COVERAGE_AUDIT*.md claimed 479/479 symbols; 353 of them are
  method-less opaque handles and 29 raw-value newtypes. They now report 97
  verified symbols and list the rest as gaps. The README states the platform
  requirements and shows versioned dependencies.

### Changed

- **BREAKING:** requires `apple-metal >=0.10, <0.11`, which declares `links` so a
  build holds a single apple-metal release, and Rust 1.82 (was 1.76).
- **BREAKING:** kernels, descriptors with setters, `NNImageNode`,
  `MpsCommandBuffer`, `State` and `StateResourceList` are `Send` but no longer
  `Sync`: MPS kernels may only be used by one thread at a time, and the setters
  and encode methods take `&self`. `Matrix`, `Vector`, `NDArray`, `Image`,
  `Predicate`, `PreferredDevice`, graph filter nodes, convolution weight states
  and recurrent image states stay `Sync`.
- **BREAKING:** these now return `Result`: `Matrix::new_with_buffer`,
  `Vector::new_with_buffer`, `NDArray::new_with_buffer`,
  `MatrixMultiplication::encode`,
  `NDArrayMatrixMultiplication::encode_to_destination`,
  `NDArrayDescriptor::set_number_of_dimensions`, `reshape_with_dimension_sizes`
  and `transpose_dimension`, `CnnConvolutionDescriptor::set_groups`,
  `CnnConvolution::new` and `encode_image`,
  `CnnConvolutionWeightsAndBiasesState::new_with_offsets` and `new_with_device`,
  the optimizer `encode_*` methods, `PolygonAccelerationStructure::rebuild` and
  `encode_refit`, `RayIntersector::encode_intersection`, and
  `ImageHistogram::encode_image` and `encode_texture`.
- **BREAKING:** the filter `encode_image` and `encode_texture` methods (every
  unary filter, `ImageAdd` and `ImageScaleAndAdd`) return `Result<()>`, and
  their textures need `texture_usage::SHADER_READ` (sources) and
  `texture_usage::SHADER_WRITE` (destination).
- **BREAKING:** `hint_temporary_memory_high_water_mark`,
  `set_heap_cache_duration`, `State::synchronize_on_command_buffer`,
  `state_batch_synchronize`, `image::image_batch_synchronize` and
  `MpsCommandBuffer::prefetch_heap_for_workload_size` return `Result<()>`, and
  `ImageHistogram::histogram_size_for_source_format` returns `Result<usize>`.
- **BREAKING:** `PolygonAccelerationStructure::set_index_buffer` is `unsafe`:
  MPS reads vertices through the indices, whose values the crate can't check.
- **BREAKING:** `Error` is `#[non_exhaustive]` and gains `BufferTooSmall`,
  `Misaligned`, `DimensionMismatch`, `InvalidArgument`, `UnsupportedDataType`,
  `Overflow`, `Unsupported`, `Rejected`, `CommandBuffer` and
  `UnsupportedPixelFormat`. `CommandBuffer` carries the
  `apple_metal::CommandBufferError` (`InvalidState` for a buffer that no longer
  records, `ActiveEncoder` for an open encoder) and is the error's `source()`.
- **BREAKING:** `State::set_read_count` returns `Result<()>`,
  `state_batch_increment_read_count` returns `Result<usize>`, and
  `RnnImageInferenceLayer::set_recurrent_output_is_temporary` returns `Result<()>`
  and refuses `true`. `State::synchronize_on_command_buffer` and
  `state_batch_synchronize` refuse temporary states. `NDArrayMatrixMultiplication::encode`
  returns regular arrays that keep their storage until dropped, instead of temporary
  arrays recycled within the command buffer. The raw `ffi::mps_state_set_read_count`,
  `mps_state_synchronize_on_command_buffer` and `mps_state_batch_synchronize` return
  `bool`, `mps_state_batch_increment_read_count` returns `isize` (-1 when refused),
  and `ffi::mps_state_release` is new.
- `NDArrayMatrixMultiplication::new` accepts 2 or 3 sources only, and the
  `ffi` image-transfer, histogram, filter-encode, heap-hint and prefetch
  functions changed signature; `ffi::mps_command_buffer_from_command_queue` is
  gone because `MpsCommandBuffer::from_command_queue` now creates the buffer
  through apple-metal.

### Added

- `MpsCommandBuffer::command_buffer` returns the wrapped apple-metal command
  buffer, so buffers made by `from_command_queue` can be committed.
- `MatrixDescriptor`, `VectorDescriptor` and `NDArrayDescriptor`
  `required_buffer_length`, `MatrixMultiplication::descriptor`,
  `NDArrayMatrixMultiplication::source_count`, `Image::feature_channel_format`
  and `feature_channel_format_size`.
- `data_type` constants for bfloat16, the complex types, 64-bit integers, bool
  and the sub-byte types; `data_type_size` covers the byte-sized ones.

### Removed

- **BREAKING:** `MpsCommandBuffer::commit_and_continue`. It committed the
  wrapped command buffer without apple-metal knowing, so the next
  `CommandBuffer::commit` aborted; it also aborted on committed buffers and
  with an open encoder.

## [0.2.6] - 2026-06-06

- `CnnConvolution::new` checks the kernel-weight and bias slice lengths against
  the descriptor; removed the unused Swift bridge C header.

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
