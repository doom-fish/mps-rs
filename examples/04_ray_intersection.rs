use apple_metal::{resource_options, MetalBuffer, MetalDevice};
use apple_mps::{
    acceleration_structure_status, cull_mode, data_type, intersection_data_type, intersection_type,
    polygon_type, ray_data_type, PolygonAccelerationStructure, RayIntersector, SVGF,
};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PackedFloat3 {
    x: f32,
    y: f32,
    z: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct PackedRayOriginDirection {
    origin: PackedFloat3,
    direction: PackedFloat3,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct IntersectionDistancePrimitiveIndex {
    distance: f32,
    primitive_index: u32,
}

fn as_bytes<T>(values: &[T]) -> &[u8] {
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), core::mem::size_of_val(values))
    }
}

fn read_struct<T: Copy>(buffer: &MetalBuffer) -> T {
    let ptr = buffer.contents().expect("buffer contents").cast::<T>();
    unsafe { *ptr }
}

#[allow(clippy::too_many_lines)]
fn main() {
    let device = MetalDevice::system_default().expect("no Metal device available");
    let queue = device.new_command_queue().expect("command queue");

    let vertices = [
        [-1.0_f32, -1.0, 0.0, 0.0],
        [1.0_f32, -1.0, 0.0, 0.0],
        [0.0_f32, 1.0, 0.0, 0.0],
    ];
    let vertex_buffer = device
        .new_buffer(
            core::mem::size_of_val(&vertices),
            resource_options::STORAGE_MODE_SHARED,
        )
        .expect("vertex buffer");
    let _ = vertex_buffer.write_bytes(as_bytes(&vertices));

    let acceleration_structure =
        PolygonAccelerationStructure::new(&device).expect("polygon acceleration structure");
    acceleration_structure.set_polygon_type(polygon_type::TRIANGLE);
    acceleration_structure.set_vertex_stride(core::mem::size_of::<[f32; 4]>());
    acceleration_structure.set_index_type(data_type::UINT32);
    acceleration_structure.set_vertex_buffer(Some(&vertex_buffer));
    acceleration_structure.set_index_buffer(None);
    acceleration_structure.set_polygon_count(1);
    acceleration_structure.rebuild();
    assert_eq!(
        acceleration_structure.status(),
        acceleration_structure_status::BUILT,
        "acceleration structure should build successfully"
    );

    let ray = PackedRayOriginDirection {
        origin: PackedFloat3 {
            x: 0.0,
            y: 0.0,
            z: 1.0,
        },
        direction: PackedFloat3 {
            x: 0.0,
            y: 0.0,
            z: -1.0,
        },
    };
    let miss = IntersectionDistancePrimitiveIndex {
        distance: -1.0,
        primitive_index: u32::MAX,
    };

    let ray_buffer = device
        .new_buffer(
            core::mem::size_of::<PackedRayOriginDirection>(),
            resource_options::STORAGE_MODE_SHARED,
        )
        .expect("ray buffer");
    let intersection_buffer = device
        .new_buffer(
            core::mem::size_of::<IntersectionDistancePrimitiveIndex>(),
            resource_options::STORAGE_MODE_SHARED,
        )
        .expect("intersection buffer");
    let _ = ray_buffer.write_bytes(as_bytes(&[ray]));
    let _ = intersection_buffer.write_bytes(as_bytes(&[miss]));

    let intersector = RayIntersector::new(&device).expect("ray intersector");
    intersector.set_cull_mode(cull_mode::NONE);
    intersector.set_ray_data_type(ray_data_type::PACKED_ORIGIN_DIRECTION);
    intersector.set_intersection_data_type(intersection_data_type::DISTANCE_PRIMITIVE_INDEX);
    assert!(
        intersector.recommended_minimum_ray_batch_size(1) >= 1,
        "recommended ray batch size should be positive"
    );

    let command_buffer = queue.new_command_buffer().expect("command buffer");
    intersector.encode_intersection(
        &command_buffer,
        intersection_type::NEAREST,
        &ray_buffer,
        0,
        &intersection_buffer,
        0,
        1,
        &acceleration_structure,
    );
    command_buffer.commit();
    command_buffer.wait_until_completed();

    let intersection = read_struct::<IntersectionDistancePrimitiveIndex>(&intersection_buffer);
    assert!(
        (intersection.distance - 1.0).abs() < 1.0e-4,
        "unexpected hit distance: {intersection:?}"
    );
    assert_eq!(
        intersection.primitive_index, 0,
        "expected to hit triangle 0"
    );

    let svgf = SVGF::new(&device).expect("svgf");
    svgf.set_depth_weight(0.75);
    svgf.set_normal_weight(64.0);
    svgf.set_luminance_weight(2.5);
    svgf.set_channel_count(3);
    svgf.set_channel_count2(1);
    assert!((svgf.depth_weight() - 0.75).abs() < f32::EPSILON);
    assert!((svgf.normal_weight() - 64.0).abs() < f32::EPSILON);
    assert!((svgf.luminance_weight() - 2.5).abs() < f32::EPSILON);
    assert_eq!(svgf.channel_count(), 3);
    assert_eq!(svgf.channel_count2(), 1);

    println!(
        "ray smoke passed: distance={:.3} primitive_index={} batch_size={}",
        intersection.distance,
        intersection.primitive_index,
        intersector.recommended_minimum_ray_batch_size(1)
    );
}
