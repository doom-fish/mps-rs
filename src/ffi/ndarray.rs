use core::ffi::c_void;

extern "C" {
    pub fn mps_ndarray_descriptor_new_with_dimension_sizes(
        data_type: u32,
        number_of_dimensions: usize,
        dimension_sizes: *const usize,
    ) -> *mut c_void;
    pub fn mps_ndarray_descriptor_data_type(handle: *mut c_void) -> u32;
    pub fn mps_ndarray_descriptor_set_data_type(handle: *mut c_void, data_type: u32);
    pub fn mps_ndarray_descriptor_number_of_dimensions(handle: *mut c_void) -> usize;
    pub fn mps_ndarray_descriptor_set_number_of_dimensions(
        handle: *mut c_void,
        number_of_dimensions: usize,
    );
    pub fn mps_ndarray_descriptor_length_of_dimension(
        handle: *mut c_void,
        dimension_index: usize,
    ) -> usize;
    pub fn mps_ndarray_descriptor_reshape_with_dimension_sizes(
        handle: *mut c_void,
        number_of_dimensions: usize,
        dimension_sizes: *const usize,
    );
    pub fn mps_ndarray_descriptor_transpose_dimension(
        handle: *mut c_void,
        dimension_index: usize,
        other_dimension_index: usize,
    );

    pub fn mps_ndarray_new_with_descriptor(
        device_handle: *mut c_void,
        descriptor_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mps_ndarray_new_scalar(device_handle: *mut c_void, value: f64) -> *mut c_void;
    pub fn mps_ndarray_new_with_buffer(
        buffer_handle: *mut c_void,
        offset: usize,
        descriptor_handle: *mut c_void,
    ) -> *mut c_void;
    pub fn mps_ndarray_data_type(handle: *mut c_void) -> u32;
    pub fn mps_ndarray_number_of_dimensions(handle: *mut c_void) -> usize;
    pub fn mps_ndarray_length_of_dimension(handle: *mut c_void, dimension_index: usize) -> usize;
    pub fn mps_ndarray_descriptor(handle: *mut c_void) -> *mut c_void;
    pub fn mps_ndarray_resource_size(handle: *mut c_void) -> usize;

    pub fn mps_ndarray_identity_new(device_handle: *mut c_void) -> *mut c_void;
    pub fn mps_ndarray_identity_reshape(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        source_array_handle: *mut c_void,
        number_of_dimensions: usize,
        dimension_sizes: *const usize,
        destination_array_handle: *mut c_void,
    ) -> *mut c_void;
}
