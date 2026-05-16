use core::ffi::c_void;

extern "C" {
    pub fn mps_matrix_descriptor_row_bytes_for_columns(columns: usize, data_type: u32) -> usize;
    pub fn mps_vector_descriptor_vector_bytes_for_length(length: usize, data_type: u32) -> usize;
    pub fn mps_matrix_new_with_buffer(
        buffer_handle: *mut c_void,
        rows: usize,
        columns: usize,
        matrices: usize,
        row_bytes: usize,
        matrix_bytes: usize,
        data_type: u32,
    ) -> *mut c_void;
    pub fn mps_matrix_rows(handle: *mut c_void) -> usize;
    pub fn mps_matrix_columns(handle: *mut c_void) -> usize;
    pub fn mps_matrix_matrices(handle: *mut c_void) -> usize;
    pub fn mps_matrix_row_bytes(handle: *mut c_void) -> usize;
    pub fn mps_matrix_matrix_bytes(handle: *mut c_void) -> usize;
    pub fn mps_matrix_data_type(handle: *mut c_void) -> u32;

    pub fn mps_vector_new_with_buffer(
        buffer_handle: *mut c_void,
        length: usize,
        vectors: usize,
        vector_bytes: usize,
        data_type: u32,
    ) -> *mut c_void;
    pub fn mps_vector_length(handle: *mut c_void) -> usize;
    pub fn mps_vector_vectors(handle: *mut c_void) -> usize;
    pub fn mps_vector_vector_bytes(handle: *mut c_void) -> usize;
    pub fn mps_vector_data_type(handle: *mut c_void) -> u32;

    pub fn mps_matrix_multiplication_new(
        device_handle: *mut c_void,
        transpose_left: bool,
        transpose_right: bool,
        result_rows: usize,
        result_columns: usize,
        interior_columns: usize,
        alpha: f64,
        beta: f64,
    ) -> *mut c_void;
    pub fn mps_matrix_multiplication_encode(
        handle: *mut c_void,
        command_buffer_handle: *mut c_void,
        left_matrix_handle: *mut c_void,
        right_matrix_handle: *mut c_void,
        result_matrix_handle: *mut c_void,
    );
}
