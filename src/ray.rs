use crate::ffi;
use apple_metal::{CommandBuffer, MetalBuffer, MetalDevice};
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

        // SAFETY: MPS handles are opaque pointers to thread-safe Swift/ObjC objects.
        unsafe impl Send for $name {}
        // SAFETY: MPS handles are opaque pointers to thread-safe Swift/ObjC objects.
        unsafe impl Sync for $name {}

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

opaque_handle!(PolygonAccelerationStructure, "Wraps `MPSPolygonAccelerationStructure`.");
impl PolygonAccelerationStructure {
    /// Wraps a constructor on `MPSPolygonAccelerationStructure`.
    #[must_use]
    pub fn new(device: &MetalDevice) -> Option<Self> {
        let ptr = unsafe { ffi::mps_polygon_acceleration_structure_new(device.as_ptr()) };
        if ptr.is_null() {
            None
        } else {
            Some(Self { ptr })
        }
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
    pub fn set_index_buffer(&self, buffer: Option<&MetalBuffer>) {
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
    pub fn rebuild(&self) {
        unsafe { ffi::mps_polygon_acceleration_structure_rebuild(self.ptr) };
    }

    /// Wraps the corresponding `MPSPolygonAccelerationStructure` encode entry point.
    pub fn encode_refit(&self, command_buffer: &CommandBuffer) {
        unsafe {
            ffi::mps_polygon_acceleration_structure_encode_refit(self.ptr, command_buffer.as_ptr());
        };
    }
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
    ) {
        unsafe {
            ffi::mps_ray_intersector_encode_intersection(
                self.ptr,
                command_buffer.as_ptr(),
                intersection_type,
                ray_buffer.as_ptr(),
                ray_buffer_offset,
                intersection_buffer.as_ptr(),
                intersection_buffer_offset,
                ray_count,
                acceleration_structure.as_ptr(),
            );
        };
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
