use crate::error::{Error, Result};
use crate::ffi;
use crate::matrix::data_type;
use apple_metal::{CommandBuffer, MetalBuffer, MetalDevice};
use core::cell::Cell;
use core::ffi::c_void;
use core::ptr;

/// `MPSPolygonType` constants.
pub mod polygon_type {
    /// Wraps a `MPSPolygonType` raw value.
    pub const TRIANGLE: usize = 0;
    /// Wraps a `MPSPolygonType` raw value.
    pub const QUADRILATERAL: usize = 1;
}

/// `MPSAccelerationStructureUsage` bitflags.
pub mod acceleration_structure_usage {
    /// Wraps a `MPSAccelerationStructureUsage` raw value.
    pub const NONE: usize = 0;
    /// Wraps a `MPSAccelerationStructureUsage` raw value.
    pub const REFIT: usize = 1;
    /// Wraps a `MPSAccelerationStructureUsage` raw value.
    pub const FREQUENT_REBUILD: usize = 2;
    /// Wraps a `MPSAccelerationStructureUsage` raw value.
    pub const PREFER_GPU_BUILD: usize = 4;
    /// Wraps a `MPSAccelerationStructureUsage` raw value.
    pub const PREFER_CPU_BUILD: usize = 8;
}

/// `MPSAccelerationStructureStatus` constants.
pub mod acceleration_structure_status {
    /// Wraps a `MPSAccelerationStructureStatus` raw value.
    pub const UNBUILT: usize = 0;
    /// Wraps a `MPSAccelerationStructureStatus` raw value.
    pub const BUILT: usize = 1;
}

/// `MPSIntersectionType` constants.
pub mod intersection_type {
    /// Wraps a `MPSIntersectionType` raw value.
    pub const NEAREST: usize = 0;
    /// Wraps a `MPSIntersectionType` raw value.
    pub const ANY: usize = 1;
}

/// `MPSRayDataType` constants.
pub mod ray_data_type {
    /// Wraps a `MPSRayDataType` raw value.
    pub const ORIGIN_DIRECTION: usize = 0;
    /// Wraps a `MPSRayDataType` raw value.
    pub const ORIGIN_MIN_DISTANCE_DIRECTION_MAX_DISTANCE: usize = 1;
    /// Wraps a `MPSRayDataType` raw value.
    pub const ORIGIN_MASK_DIRECTION_MAX_DISTANCE: usize = 2;
    /// Wraps a `MPSRayDataType` raw value.
    pub const PACKED_ORIGIN_DIRECTION: usize = 3;
}

/// `MPSIntersectionDataType` constants.
pub mod intersection_data_type {
    /// Wraps a `MPSIntersectionDataType` raw value.
    pub const DISTANCE: usize = 0;
    /// Wraps a `MPSIntersectionDataType` raw value.
    pub const DISTANCE_PRIMITIVE_INDEX: usize = 1;
    /// Wraps a `MPSIntersectionDataType` raw value.
    pub const DISTANCE_PRIMITIVE_INDEX_COORDINATES: usize = 2;
    /// Wraps a `MPSIntersectionDataType` raw value.
    pub const DISTANCE_PRIMITIVE_INDEX_INSTANCE_INDEX: usize = 3;
    /// Wraps a `MPSIntersectionDataType` raw value.
    pub const DISTANCE_PRIMITIVE_INDEX_INSTANCE_INDEX_COORDINATES: usize = 4;
    /// Wraps a `MPSIntersectionDataType` raw value.
    pub const DISTANCE_PRIMITIVE_INDEX_BUFFER_INDEX: usize = 5;
    /// Wraps a `MPSIntersectionDataType` raw value.
    pub const DISTANCE_PRIMITIVE_INDEX_BUFFER_INDEX_COORDINATES: usize = 6;
    /// Wraps a `MPSIntersectionDataType` raw value.
    pub const DISTANCE_PRIMITIVE_INDEX_BUFFER_INDEX_INSTANCE_INDEX: usize = 7;
    /// Wraps a `MPSIntersectionDataType` raw value.
    pub const DISTANCE_PRIMITIVE_INDEX_BUFFER_INDEX_INSTANCE_INDEX_COORDINATES: usize = 8;
}

/// `MTLCullMode` constants.
pub mod cull_mode {
    /// Wraps a `MTLCullMode` raw value.
    pub const NONE: usize = 0;
    /// Wraps a `MTLCullMode` raw value.
    pub const FRONT: usize = 1;
    /// Wraps a `MTLCullMode` raw value.
    pub const BACK: usize = 2;
}

/// `MTLWinding` constants.
pub mod winding {
    /// Wraps a `MTLWinding` raw value.
    pub const CLOCKWISE: usize = 0;
    /// Wraps a `MTLWinding` raw value.
    pub const COUNTER_CLOCKWISE: usize = 1;
}

#[doc(hidden)]
pub use crate::generated::ray::*;

macro_rules! opaque_handle {
    ($name:ident, $doc:expr) => {
        #[doc = $doc]
        pub struct $name {
            ptr: *mut c_void,
        }

        // SAFETY: MPS kernels may move between threads; only one thread may use one at a time.
        unsafe impl Send for $name {}

        impl Drop for $name {
            fn drop(&mut self) {
                if !self.ptr.is_null() {
                    // SAFETY: `ptr` is a +1 retained MPS object owned by this wrapper.
                    unsafe { ffi::mps_object_release(self.ptr) };
                    self.ptr = ptr::null_mut();
                }
            }
        }

        impl $name {
            /// Returns the retained Objective-C pointer backing this wrapper.
            #[must_use]
            pub const fn as_ptr(&self) -> *mut c_void {
                self.ptr
            }
        }
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Geometry {
    polygon_type: usize,
    vertex_stride: usize,
    index_type: Option<u32>,
    polygon_count: usize,
    usage: usize,
}

/// Wraps `MPSPolygonAccelerationStructure`.
pub struct PolygonAccelerationStructure {
    ptr: *mut c_void,
    built: Cell<Option<Geometry>>,
}

unsafe impl Send for PolygonAccelerationStructure {}

impl Drop for PolygonAccelerationStructure {
    fn drop(&mut self) {
        if !self.ptr.is_null() {
            // SAFETY: `ptr` is a +1 retained MPS object owned by this wrapper.
            unsafe { ffi::mps_object_release(self.ptr) };
            self.ptr = ptr::null_mut();
        }
    }
}

impl PolygonAccelerationStructure {
    /// Wraps a constructor on `MPSPolygonAccelerationStructure`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        let ptr = unsafe { ffi::mps_polygon_acceleration_structure_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self {
                ptr,
                built: Cell::new(None),
            })
        }
    }

    /// Returns the retained Objective-C pointer backing this wrapper.
    #[must_use]
    pub const fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` method.
    #[must_use]
    pub fn polygon_type(&self) -> usize {
        unsafe { ffi::mps_polygon_acceleration_structure_polygon_type(self.ptr) }
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` setter.
    pub fn set_polygon_type(&self, polygon_type: usize) {
        unsafe { ffi::mps_polygon_acceleration_structure_set_polygon_type(self.ptr, polygon_type) };
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` method.
    #[must_use]
    pub fn vertex_stride(&self) -> usize {
        unsafe { ffi::mps_polygon_acceleration_structure_vertex_stride(self.ptr) }
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` setter.
    pub fn set_vertex_stride(&self, vertex_stride: usize) {
        unsafe {
            ffi::mps_polygon_acceleration_structure_set_vertex_stride(self.ptr, vertex_stride);
        };
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` method.
    #[must_use]
    pub fn index_type(&self) -> u32 {
        unsafe { ffi::mps_polygon_acceleration_structure_index_type(self.ptr) }
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` setter.
    pub fn set_index_type(&self, index_type: u32) {
        unsafe { ffi::mps_polygon_acceleration_structure_set_index_type(self.ptr, index_type) };
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` setter.
    pub fn set_vertex_buffer(&self, buffer: Option<&MetalBuffer>) {
        let buffer_ptr = buffer.map_or(ptr::null_mut(), MetalBuffer::as_ptr);
        unsafe { ffi::mps_polygon_acceleration_structure_set_vertex_buffer(self.ptr, buffer_ptr) };
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` method.
    #[must_use]
    pub fn vertex_buffer_offset(&self) -> usize {
        unsafe { ffi::mps_polygon_acceleration_structure_vertex_buffer_offset(self.ptr) }
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` setter.
    pub fn set_vertex_buffer_offset(&self, offset: usize) {
        unsafe {
            ffi::mps_polygon_acceleration_structure_set_vertex_buffer_offset(self.ptr, offset);
        };
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` setter.
    #[allow(clippy::missing_safety_doc)]
    pub unsafe fn set_index_buffer(&self, buffer: Option<&MetalBuffer>) {
        let buffer_ptr = buffer.map_or(ptr::null_mut(), MetalBuffer::as_ptr);
        unsafe { ffi::mps_polygon_acceleration_structure_set_index_buffer(self.ptr, buffer_ptr) };
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` method.
    #[must_use]
    pub fn index_buffer_offset(&self) -> usize {
        unsafe { ffi::mps_polygon_acceleration_structure_index_buffer_offset(self.ptr) }
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` setter.
    pub fn set_index_buffer_offset(&self, offset: usize) {
        unsafe {
            ffi::mps_polygon_acceleration_structure_set_index_buffer_offset(self.ptr, offset);
        };
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` method.
    #[must_use]
    pub fn polygon_count(&self) -> usize {
        unsafe { ffi::mps_polygon_acceleration_structure_polygon_count(self.ptr) }
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` setter.
    pub fn set_polygon_count(&self, count: usize) {
        unsafe { ffi::mps_polygon_acceleration_structure_set_polygon_count(self.ptr, count) };
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` method.
    #[must_use]
    pub fn usage(&self) -> usize {
        unsafe { ffi::mps_polygon_acceleration_structure_usage(self.ptr) }
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` setter.
    pub fn set_usage(&self, usage: usize) {
        unsafe { ffi::mps_polygon_acceleration_structure_set_usage(self.ptr, usage) };
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` method.
    #[must_use]
    pub fn status(&self) -> usize {
        unsafe { ffi::mps_polygon_acceleration_structure_status(self.ptr) }
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` method.
    pub fn rebuild(&self) -> Result<()> {
        let geometry = self.validated_geometry()?;
        self.built.set(None);
        unsafe { ffi::mps_polygon_acceleration_structure_rebuild(self.ptr) };
        if self.status() == acceleration_structure_status::BUILT {
            self.built.set(Some(geometry));
            Ok(())
        } else {
            Err(Error::Rejected("MPSAccelerationStructure rebuild"))
        }
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` encode entry point.
    pub fn encode_refit(&self, command_buffer: &CommandBuffer) -> Result<()> {
        if self.ensure_built()?.usage & acceleration_structure_usage::REFIT == 0 {
            return Err(Error::InvalidArgument(
                "refitting needs a structure rebuilt with acceleration_structure_usage::REFIT",
            ));
        }
        crate::core::encode(command_buffer, |buffer| unsafe {
            ffi::mps_polygon_acceleration_structure_encode_refit(self.ptr, buffer);
        })
    }

    fn ensure_built(&self) -> Result<Geometry> {
        let built = self
            .built
            .get()
            .filter(|_| self.status() == acceleration_structure_status::BUILT)
            .ok_or(Error::InvalidArgument(
                "the acceleration structure must be rebuilt first",
            ))?;
        if self.validated_geometry()? == built {
            Ok(built)
        } else {
            Err(Error::InvalidArgument(
                "polygon type, count, stride, index type or usage changed since the last rebuild",
            ))
        }
    }

    fn validated_index_type(&self, corners: usize) -> Result<Option<u32>> {
        let index_length = native_length(unsafe {
            ffi::mps_polygon_acceleration_structure_index_buffer_length(self.ptr)
        });
        match index_length {
            Some(index_length) => {
                let index_type = self.index_type();
                let index_size = match index_type {
                    data_type::UINT16 => 2,
                    data_type::UINT32 => 4,
                    _ => {
                        return Err(Error::InvalidArgument(
                            "index_type must be UINT16 or UINT32",
                        ))
                    }
                };
                let index_offset = self.index_buffer_offset();
                if index_offset % index_size != 0 {
                    return Err(Error::Misaligned {
                        field: "index_buffer_offset",
                        value: index_offset,
                        alignment: index_size,
                    });
                }
                let required = corners
                    .checked_mul(index_size)
                    .and_then(|bytes| bytes.checked_add(index_offset))
                    .ok_or(Error::Overflow)?;
                if required > index_length {
                    return Err(Error::BufferTooSmall {
                        required,
                        length: index_length,
                    });
                }
                Ok(Some(index_type))
            }
            None => Ok(None),
        }
    }

    fn validated_geometry(&self) -> Result<Geometry> {
        let polygon_type = self.polygon_type();
        let vertices_per_polygon = match polygon_type {
            polygon_type::TRIANGLE => 3,
            polygon_type::QUADRILATERAL => 4,
            _ => {
                return Err(Error::InvalidArgument(
                    "polygon_type is not an MPSPolygonType",
                ))
            }
        };
        let vertex_stride = match self.vertex_stride() {
            0 => DEFAULT_VERTEX_STRIDE,
            stride => stride,
        };
        if vertex_stride % 4 != 0 {
            return Err(Error::Misaligned {
                field: "vertex_stride",
                value: vertex_stride,
                alignment: 4,
            });
        }
        if vertex_stride < VERTEX_SIZE {
            return Err(Error::DimensionMismatch {
                field: "vertex_stride",
                expected: VERTEX_SIZE,
                actual: vertex_stride,
            });
        }
        let vertex_offset = self.vertex_buffer_offset();
        if vertex_offset % 4 != 0 {
            return Err(Error::Misaligned {
                field: "vertex_buffer_offset",
                value: vertex_offset,
                alignment: 4,
            });
        }
        let vertex_length = native_length(unsafe {
            ffi::mps_polygon_acceleration_structure_vertex_buffer_length(self.ptr)
        })
        .ok_or(Error::InvalidArgument("a vertex buffer is required"))?;
        let polygon_count = self.polygon_count();
        let corners = polygon_count
            .checked_mul(vertices_per_polygon)
            .ok_or(Error::Overflow)?;
        let index_type = self.validated_index_type(corners)?;
        let vertices = if index_type.is_some() {
            usize::from(polygon_count > 0)
        } else {
            corners
        };
        if vertices > 0 {
            let required = (vertices - 1)
                .checked_mul(vertex_stride)
                .and_then(|bytes| bytes.checked_add(VERTEX_SIZE))
                .and_then(|bytes| bytes.checked_add(vertex_offset))
                .ok_or(Error::Overflow)?;
            if required > vertex_length {
                return Err(Error::BufferTooSmall {
                    required,
                    length: vertex_length,
                });
            }
        }
        Ok(Geometry {
            polygon_type,
            vertex_stride,
            index_type,
            polygon_count,
            usage: self.usage(),
        })
    }
}

const VERTEX_SIZE: usize = 12;
const DEFAULT_VERTEX_STRIDE: usize = 16;

fn native_length(length: isize) -> Option<usize> {
    usize::try_from(length).ok()
}

const fn ray_layout(ray_data_type: usize) -> Option<(usize, usize)> {
    match ray_data_type {
        ray_data_type::ORIGIN_DIRECTION => Some((32, 16)),
        ray_data_type::ORIGIN_MIN_DISTANCE_DIRECTION_MAX_DISTANCE
        | ray_data_type::ORIGIN_MASK_DIRECTION_MAX_DISTANCE => Some((32, 4)),
        ray_data_type::PACKED_ORIGIN_DIRECTION => Some((24, 4)),
        _ => None,
    }
}

const fn intersection_layout(intersection_data_type: usize) -> Option<(usize, usize)> {
    match intersection_data_type {
        intersection_data_type::DISTANCE => Some((4, 4)),
        intersection_data_type::DISTANCE_PRIMITIVE_INDEX => Some((8, 4)),
        intersection_data_type::DISTANCE_PRIMITIVE_INDEX_COORDINATES => Some((16, 8)),
        intersection_data_type::DISTANCE_PRIMITIVE_INDEX_INSTANCE_INDEX
        | intersection_data_type::DISTANCE_PRIMITIVE_INDEX_BUFFER_INDEX => Some((12, 4)),
        intersection_data_type::DISTANCE_PRIMITIVE_INDEX_INSTANCE_INDEX_COORDINATES
        | intersection_data_type::DISTANCE_PRIMITIVE_INDEX_BUFFER_INDEX_COORDINATES
        | intersection_data_type::DISTANCE_PRIMITIVE_INDEX_BUFFER_INDEX_INSTANCE_INDEX_COORDINATES => {
            Some((24, 8))
        }
        intersection_data_type::DISTANCE_PRIMITIVE_INDEX_BUFFER_INDEX_INSTANCE_INDEX => {
            Some((16, 4))
        }
        _ => None,
    }
}

fn effective_stride(
    field: &'static str,
    stride: usize,
    (size, alignment): (usize, usize),
) -> Result<usize> {
    if stride == 0 {
        return Ok(size);
    }
    if stride % alignment != 0 {
        return Err(Error::Misaligned {
            field,
            value: stride,
            alignment,
        });
    }
    if stride < size {
        return Err(Error::DimensionMismatch {
            field,
            expected: size,
            actual: stride,
        });
    }
    Ok(stride)
}

fn ensure_records(
    field: &'static str,
    buffer: &MetalBuffer,
    offset: usize,
    stride: usize,
    size: usize,
    count: usize,
) -> Result<()> {
    if offset % stride != 0 {
        return Err(Error::Misaligned {
            field,
            value: offset,
            alignment: stride,
        });
    }
    let required = match count {
        0 => offset,
        count => (count - 1)
            .checked_mul(stride)
            .and_then(|bytes| bytes.checked_add(size))
            .and_then(|bytes| bytes.checked_add(offset))
            .ok_or(Error::Overflow)?,
    };
    let length = buffer.length();
    if required > length {
        return Err(Error::BufferTooSmall { required, length });
    }
    Ok(())
}

opaque_handle!(RayIntersector, "Wraps `MPSRayIntersector`.");
impl RayIntersector {
    /// Wraps a constructor on `MPSRayIntersector`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        let ptr = unsafe { ffi::mps_ray_intersector_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps the corresponding `MPSRayIntersector` method.
    #[must_use]
    pub fn cull_mode(&self) -> usize {
        unsafe { ffi::mps_ray_intersector_cull_mode(self.ptr) }
    }

    /// Wraps the corresponding `MPSRayIntersector` setter.
    pub fn set_cull_mode(&self, cull_mode: usize) {
        unsafe { ffi::mps_ray_intersector_set_cull_mode(self.ptr, cull_mode) };
    }

    /// Wraps the corresponding `MPSRayIntersector` method.
    #[must_use]
    pub fn front_facing_winding(&self) -> usize {
        unsafe { ffi::mps_ray_intersector_front_facing_winding(self.ptr) }
    }

    /// Wraps the corresponding `MPSRayIntersector` setter.
    pub fn set_front_facing_winding(&self, winding: usize) {
        unsafe { ffi::mps_ray_intersector_set_front_facing_winding(self.ptr, winding) };
    }

    /// Wraps the corresponding `MPSRayIntersector` method.
    #[must_use]
    pub fn ray_stride(&self) -> usize {
        unsafe { ffi::mps_ray_intersector_ray_stride(self.ptr) }
    }

    /// Wraps the corresponding `MPSRayIntersector` setter.
    pub fn set_ray_stride(&self, stride: usize) {
        unsafe { ffi::mps_ray_intersector_set_ray_stride(self.ptr, stride) };
    }

    /// Wraps the corresponding `MPSRayIntersector` method.
    #[must_use]
    pub fn intersection_stride(&self) -> usize {
        unsafe { ffi::mps_ray_intersector_intersection_stride(self.ptr) }
    }

    /// Wraps the corresponding `MPSRayIntersector` setter.
    pub fn set_intersection_stride(&self, stride: usize) {
        unsafe { ffi::mps_ray_intersector_set_intersection_stride(self.ptr, stride) };
    }

    /// Wraps the corresponding `MPSRayIntersector` method.
    #[must_use]
    pub fn ray_data_type(&self) -> usize {
        unsafe { ffi::mps_ray_intersector_ray_data_type(self.ptr) }
    }

    /// Wraps the corresponding `MPSRayIntersector` setter.
    pub fn set_ray_data_type(&self, data_type: usize) {
        unsafe { ffi::mps_ray_intersector_set_ray_data_type(self.ptr, data_type) };
    }

    /// Wraps the corresponding `MPSRayIntersector` method.
    #[must_use]
    pub fn intersection_data_type(&self) -> usize {
        unsafe { ffi::mps_ray_intersector_intersection_data_type(self.ptr) }
    }

    /// Wraps the corresponding `MPSRayIntersector` setter.
    pub fn set_intersection_data_type(&self, data_type: usize) {
        unsafe { ffi::mps_ray_intersector_set_intersection_data_type(self.ptr, data_type) };
    }

    /// Wraps the corresponding `MPSRayIntersector` method.
    #[must_use]
    pub fn recommended_minimum_ray_batch_size(&self, ray_count: usize) -> usize {
        unsafe { ffi::mps_ray_intersector_recommended_minimum_ray_batch_size(self.ptr, ray_count) }
    }

    /// Wraps the corresponding `MPSRayIntersector` encode entry point.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_intersection(
        &self,
        command_buffer: &CommandBuffer,
        intersection_type: usize,
        ray_buffer: &MetalBuffer,
        ray_buffer_offset: usize,
        intersection_buffer: &MetalBuffer,
        intersection_buffer_offset: usize,
        ray_count: usize,
        acceleration_structure: &PolygonAccelerationStructure,
    ) -> Result<()> {
        if intersection_type > intersection_type::ANY {
            return Err(Error::InvalidArgument(
                "intersection_type is not an MPSIntersectionType",
            ));
        }
        let ray = ray_layout(self.ray_data_type()).ok_or(Error::InvalidArgument(
            "ray_data_type is not an MPSRayDataType",
        ))?;
        let hit = intersection_layout(self.intersection_data_type()).ok_or(
            Error::InvalidArgument("intersection_data_type is not an MPSIntersectionDataType"),
        )?;
        let ray_stride = effective_stride("ray_stride", self.ray_stride(), ray)?;
        let hit_stride = effective_stride("intersection_stride", self.intersection_stride(), hit)?;
        ensure_records(
            "ray_buffer_offset",
            ray_buffer,
            ray_buffer_offset,
            ray_stride,
            ray.0,
            ray_count,
        )?;
        ensure_records(
            "intersection_buffer_offset",
            intersection_buffer,
            intersection_buffer_offset,
            hit_stride,
            hit.0,
            ray_count,
        )?;
        acceleration_structure.ensure_built()?;
        crate::core::encode(command_buffer, |buffer| unsafe {
            ffi::mps_ray_intersector_encode_intersection(
                self.ptr,
                buffer,
                intersection_type,
                ray_buffer.as_ptr(),
                ray_buffer_offset,
                intersection_buffer.as_ptr(),
                intersection_buffer_offset,
                ray_count,
                acceleration_structure.as_ptr(),
            );
        })
    }
}

opaque_handle!(SVGF, "Wraps `MPSSVGF`.");
impl SVGF {
    /// Wraps a constructor on `MPSSVGF`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        let ptr = unsafe { ffi::mps_svgf_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
    }

    /// Wraps the corresponding `MPSSVGF` method.
    #[must_use]
    pub fn depth_weight(&self) -> f32 {
        unsafe { ffi::mps_svgf_depth_weight(self.ptr) }
    }

    /// Wraps the corresponding `MPSSVGF` setter.
    pub fn set_depth_weight(&self, value: f32) {
        unsafe { ffi::mps_svgf_set_depth_weight(self.ptr, value) };
    }

    /// Wraps the corresponding `MPSSVGF` method.
    #[must_use]
    pub fn normal_weight(&self) -> f32 {
        unsafe { ffi::mps_svgf_normal_weight(self.ptr) }
    }

    /// Wraps the corresponding `MPSSVGF` setter.
    pub fn set_normal_weight(&self, value: f32) {
        unsafe { ffi::mps_svgf_set_normal_weight(self.ptr, value) };
    }

    /// Wraps the corresponding `MPSSVGF` method.
    #[must_use]
    pub fn luminance_weight(&self) -> f32 {
        unsafe { ffi::mps_svgf_luminance_weight(self.ptr) }
    }

    /// Wraps the corresponding `MPSSVGF` setter.
    pub fn set_luminance_weight(&self, value: f32) {
        unsafe { ffi::mps_svgf_set_luminance_weight(self.ptr, value) };
    }

    /// Wraps the corresponding `MPSSVGF` method.
    #[must_use]
    pub fn channel_count(&self) -> usize {
        unsafe { ffi::mps_svgf_channel_count(self.ptr) }
    }

    /// Wraps the corresponding `MPSSVGF` setter.
    pub fn set_channel_count(&self, value: usize) {
        unsafe { ffi::mps_svgf_set_channel_count(self.ptr, value) };
    }

    /// Wraps the corresponding `MPSSVGF` method.
    #[must_use]
    pub fn channel_count2(&self) -> usize {
        unsafe { ffi::mps_svgf_channel_count2(self.ptr) }
    }

    /// Wraps the corresponding `MPSSVGF` setter.
    pub fn set_channel_count2(&self, value: usize) {
        unsafe { ffi::mps_svgf_set_channel_count2(self.ptr, value) };
    }
}
