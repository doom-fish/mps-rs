use core::ffi::c_void;

extern "C" {
    pub fn mps_polygon_acceleration_structure_new(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_polygon_acceleration_structure_polygon_type(handle: *mut c_void) -> usize;
    pub fn mps_polygon_acceleration_structure_set_polygon_type(
        handle: *mut c_void,
        polygon_type: usize,
    );
    pub fn mps_polygon_acceleration_structure_vertex_stride(handle: *mut c_void) -> usize;
    pub fn mps_polygon_acceleration_structure_set_vertex_stride(
        handle: *mut c_void,
        vertex_stride: usize,
    );
    pub fn mps_polygon_acceleration_structure_index_type(handle: *mut c_void) -> u32;
    pub fn mps_polygon_acceleration_structure_set_index_type(handle: *mut c_void, index_type: u32);
    pub fn mps_polygon_acceleration_structure_set_vertex_buffer(
        handle: *mut c_void,
        buffer_handle: *mut c_void,
    );
    pub fn mps_polygon_acceleration_structure_vertex_buffer_offset(handle: *mut c_void) -> usize;
    pub fn mps_polygon_acceleration_structure_set_vertex_buffer_offset(
        handle: *mut c_void,
        offset: usize,
    );
    pub fn mps_polygon_acceleration_structure_set_index_buffer(
        handle: *mut c_void,
        buffer_handle: *mut c_void,
    );
    pub fn mps_polygon_acceleration_structure_index_buffer_offset(handle: *mut c_void) -> usize;
    pub fn mps_polygon_acceleration_structure_set_index_buffer_offset(
        handle: *mut c_void,
        offset: usize,
    );
    pub fn mps_polygon_acceleration_structure_polygon_count(handle: *mut c_void) -> usize;
    pub fn mps_polygon_acceleration_structure_set_polygon_count(handle: *mut c_void, count: usize);
    pub fn mps_polygon_acceleration_structure_usage(handle: *mut c_void) -> usize;
    pub fn mps_polygon_acceleration_structure_set_usage(handle: *mut c_void, usage: usize);
    pub fn mps_polygon_acceleration_structure_status(handle: *mut c_void) -> usize;
    pub fn mps_polygon_acceleration_structure_rebuild(handle: *mut c_void);
    pub fn mps_polygon_acceleration_structure_encode_refit(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
    );

    pub fn mps_ray_intersector_new(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_ray_intersector_cull_mode(handle: *mut c_void) -> usize;
    pub fn mps_ray_intersector_set_cull_mode(handle: *mut c_void, cull_mode: usize);
    pub fn mps_ray_intersector_front_facing_winding(handle: *mut c_void) -> usize;
    pub fn mps_ray_intersector_set_front_facing_winding(handle: *mut c_void, winding: usize);
    pub fn mps_ray_intersector_ray_stride(handle: *mut c_void) -> usize;
    pub fn mps_ray_intersector_set_ray_stride(handle: *mut c_void, stride: usize);
    pub fn mps_ray_intersector_intersection_stride(handle: *mut c_void) -> usize;
    pub fn mps_ray_intersector_set_intersection_stride(handle: *mut c_void, stride: usize);
    pub fn mps_ray_intersector_ray_data_type(handle: *mut c_void) -> usize;
    pub fn mps_ray_intersector_set_ray_data_type(handle: *mut c_void, data_type: usize);
    pub fn mps_ray_intersector_intersection_data_type(handle: *mut c_void) -> usize;
    pub fn mps_ray_intersector_set_intersection_data_type(handle: *mut c_void, data_type: usize);
    pub fn mps_ray_intersector_recommended_minimum_ray_batch_size(
        handle: *mut c_void,
        ray_count: usize,
    ) -> usize;
    pub fn mps_ray_intersector_encode_intersection(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        intersection_type: usize,
        ray_buffer_handle: *mut c_void,
        ray_buffer_offset: usize,
        intersection_buffer_handle: *mut c_void,
        intersection_buffer_offset: usize,
        ray_count: usize,
        acceleration_structure_handle: *mut c_void,
    );

    pub fn mps_svgf_new(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_svgf_depth_weight(handle: *mut c_void) -> f32;
    pub fn mps_svgf_set_depth_weight(handle: *mut c_void, value: f32);
    pub fn mps_svgf_normal_weight(handle: *mut c_void) -> f32;
    pub fn mps_svgf_set_normal_weight(handle: *mut c_void, value: f32);
    pub fn mps_svgf_luminance_weight(handle: *mut c_void) -> f32;
    pub fn mps_svgf_set_luminance_weight(handle: *mut c_void, value: f32);
    pub fn mps_svgf_channel_count(handle: *mut c_void) -> usize;
    pub fn mps_svgf_set_channel_count(handle: *mut c_void, value: usize);
    pub fn mps_svgf_channel_count2(handle: *mut c_void) -> usize;
    pub fn mps_svgf_set_channel_count2(handle: *mut c_void, value: usize);
}
