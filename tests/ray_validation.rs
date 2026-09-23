use apple_metal::{resource_options, CommandBuffer, MetalBuffer, MetalDevice};
use apple_mps::{
    acceleration_structure_usage, data_type, intersection_data_type, intersection_type,
    polygon_type, ray_data_type, Error, PolygonAccelerationStructure, RayIntersector,
};

fn device() -> MetalDevice {
    MetalDevice::system_default().expect("no Metal device available")
}

fn buffer_with(device: &MetalDevice, bytes: &[u8]) -> MetalBuffer {
    let buffer = device
        .new_buffer(bytes.len().max(4), resource_options::STORAGE_MODE_SHARED)
        .expect("buffer");
    unsafe { buffer.write_bytes(0, bytes) }.expect("write buffer");
    buffer
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect()
}

fn triangle_vertices(copies: usize) -> Vec<u8> {
    let triangle = [
        -1.0_f32, -1.0, 0.0, 0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
    ];
    f32_bytes(&triangle.repeat(copies))
}

fn structure(
    device: &MetalDevice,
    vertices: &MetalBuffer,
    count: usize,
) -> PolygonAccelerationStructure {
    let structure = PolygonAccelerationStructure::new(device).expect("acceleration structure");
    structure.set_polygon_type(polygon_type::TRIANGLE);
    structure.set_vertex_stride(16);
    structure.set_vertex_buffer(Some(vertices));
    structure.set_polygon_count(count);
    structure
}

fn packed_intersector(device: &MetalDevice) -> RayIntersector {
    let intersector = RayIntersector::new(device).expect("intersector");
    intersector.set_ray_data_type(ray_data_type::PACKED_ORIGIN_DIRECTION);
    intersector.set_intersection_data_type(intersection_data_type::DISTANCE_PRIMITIVE_INDEX);
    intersector
}

fn ray_bytes() -> Vec<u8> {
    f32_bytes(&[0.0, 0.0, 1.0, 0.0, 0.0, -1.0])
}

fn command_buffer(device: &MetalDevice) -> CommandBuffer {
    device
        .new_command_queue()
        .expect("queue")
        .new_command_buffer()
        .expect("command buffer")
}

#[test]
fn rebuild_checks_the_vertex_buffer() {
    let device = device();
    let vertices = buffer_with(&device, &triangle_vertices(1));
    let structure = structure(&device, &vertices, 2);
    assert_eq!(
        structure.rebuild(),
        Err(Error::BufferTooSmall {
            required: 92,
            length: 48,
        })
    );
    structure.set_polygon_count(1);
    structure.rebuild().expect("one triangle fits");

    structure.set_vertex_stride(13);
    assert_eq!(
        structure.rebuild(),
        Err(Error::Misaligned {
            field: "vertex_stride",
            value: 13,
            alignment: 4,
        })
    );
    structure.set_vertex_stride(8);
    assert_eq!(
        structure.rebuild(),
        Err(Error::DimensionMismatch {
            field: "vertex_stride",
            expected: 12,
            actual: 8,
        })
    );
    structure.set_vertex_stride(16);
    structure.set_vertex_buffer_offset(2);
    assert!(matches!(
        structure.rebuild(),
        Err(Error::Misaligned {
            field: "vertex_buffer_offset",
            ..
        })
    ));
    structure.set_vertex_buffer_offset(16);
    assert!(matches!(
        structure.rebuild(),
        Err(Error::BufferTooSmall { .. })
    ));
    structure.set_vertex_buffer_offset(0);
    structure.set_polygon_type(7);
    assert_eq!(structure.polygon_type(), polygon_type::TRIANGLE);
    structure.set_vertex_buffer(None);
    assert!(matches!(
        structure.rebuild(),
        Err(Error::InvalidArgument(_))
    ));
}

#[test]
fn rebuild_checks_the_index_buffer() {
    let device = device();
    let vertices = buffer_with(&device, &triangle_vertices(1));
    let structure = structure(&device, &vertices, 1);
    let short = buffer_with(&device, &[0, 0, 0, 0]);
    unsafe { structure.set_index_buffer(Some(&short)) };
    structure.set_index_type(data_type::UINT32);
    assert_eq!(
        structure.rebuild(),
        Err(Error::BufferTooSmall {
            required: 12,
            length: 4,
        })
    );
    let indices: Vec<u8> = [0_u32, 1, 2, 0]
        .iter()
        .flat_map(|index| index.to_ne_bytes())
        .collect();
    let indices = buffer_with(&device, &indices);
    unsafe { structure.set_index_buffer(Some(&indices)) };
    structure.set_index_buffer_offset(2);
    assert_eq!(
        structure.rebuild(),
        Err(Error::Misaligned {
            field: "index_buffer_offset",
            value: 2,
            alignment: 4,
        })
    );
    structure.set_index_buffer_offset(8);
    assert_eq!(
        structure.rebuild(),
        Err(Error::BufferTooSmall {
            required: 20,
            length: 16,
        })
    );
    structure.set_index_buffer_offset(0);
    structure.set_index_type(data_type::FLOAT32);
    assert_eq!(structure.index_type(), data_type::UINT32);
    structure.rebuild().expect("indexed triangle");

    let intersector = packed_intersector(&device);
    let rays = buffer_with(&device, &ray_bytes());
    let hits = buffer_with(&device, &[0xFF; 8]);
    let command_buffer = command_buffer(&device);
    intersector
        .encode_intersection(
            &command_buffer,
            intersection_type::NEAREST,
            &rays,
            0,
            &hits,
            0,
            1,
            &structure,
        )
        .expect("encode");
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
    let mut hit = [0_u8; 8];
    unsafe { hits.read_bytes(0, &mut hit) }.expect("read hit");
    let distance = f32::from_ne_bytes([hit[0], hit[1], hit[2], hit[3]]);
    assert!((distance - 1.0).abs() < 1.0e-4, "distance {distance}");
    assert_eq!(u32::from_ne_bytes([hit[4], hit[5], hit[6], hit[7]]), 0);
}

#[test]
fn encoding_requires_a_current_rebuild() {
    let device = device();
    let vertices = buffer_with(&device, &triangle_vertices(2));
    let structure = structure(&device, &vertices, 1);
    let intersector = packed_intersector(&device);
    let rays = buffer_with(&device, &ray_bytes());
    let hits = buffer_with(&device, &[0; 8]);
    let command_buffer = command_buffer(&device);
    let encode = |structure: &PolygonAccelerationStructure| {
        intersector.encode_intersection(
            &command_buffer,
            intersection_type::NEAREST,
            &rays,
            0,
            &hits,
            0,
            1,
            structure,
        )
    };
    assert!(matches!(encode(&structure), Err(Error::InvalidArgument(_))));
    assert!(matches!(
        structure.encode_refit(&command_buffer),
        Err(Error::InvalidArgument(_))
    ));
    structure.rebuild().expect("rebuild");
    structure.set_polygon_count(2);
    assert!(matches!(encode(&structure), Err(Error::InvalidArgument(_))));
    assert!(matches!(
        structure.encode_refit(&command_buffer),
        Err(Error::InvalidArgument(_))
    ));
    let smaller = buffer_with(&device, &triangle_vertices(1));
    structure.set_polygon_count(1);
    structure.set_vertex_buffer(Some(&smaller));
    encode(&structure).expect("same geometry, new buffer");
    assert!(matches!(
        structure.encode_refit(&command_buffer),
        Err(Error::InvalidArgument(_))
    ));
    structure.set_usage(acceleration_structure_usage::REFIT);
    assert!(matches!(encode(&structure), Err(Error::InvalidArgument(_))));
    structure.rebuild().expect("rebuild for refitting");
    structure.encode_refit(&command_buffer).expect("refit");
    structure.set_vertex_buffer(Some(&buffer_with(&device, &[0; 16])));
    assert!(matches!(
        encode(&structure),
        Err(Error::BufferTooSmall { .. })
    ));
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
}

#[test]
fn ray_and_intersection_buffers_are_checked() {
    let device = device();
    let vertices = buffer_with(&device, &triangle_vertices(1));
    let structure = structure(&device, &vertices, 1);
    structure.rebuild().expect("rebuild");
    let intersector = packed_intersector(&device);
    let rays = buffer_with(&device, &ray_bytes());
    let hits = buffer_with(&device, &[0; 8]);
    let command_buffer = command_buffer(&device);
    let encode = |kind: usize, ray_offset: usize, hit_offset: usize, count: usize| {
        intersector.encode_intersection(
            &command_buffer,
            kind,
            &rays,
            ray_offset,
            &hits,
            hit_offset,
            count,
            &structure,
        )
    };
    assert_eq!(
        encode(intersection_type::NEAREST, 0, 0, 2),
        Err(Error::BufferTooSmall {
            required: 48,
            length: 24,
        })
    );
    assert_eq!(
        encode(intersection_type::NEAREST, 4, 0, 1),
        Err(Error::Misaligned {
            field: "ray_buffer_offset",
            value: 4,
            alignment: 24,
        })
    );
    assert_eq!(
        encode(intersection_type::NEAREST, 0, 8, 1),
        Err(Error::BufferTooSmall {
            required: 16,
            length: 8,
        })
    );
    assert!(matches!(encode(5, 0, 0, 1), Err(Error::InvalidArgument(_))));
    intersector.set_ray_stride(20);
    assert_eq!(
        encode(intersection_type::NEAREST, 0, 0, 1),
        Err(Error::DimensionMismatch {
            field: "ray_stride",
            expected: 24,
            actual: 20,
        })
    );
    intersector.set_ray_stride(26);
    assert!(matches!(
        encode(intersection_type::NEAREST, 0, 0, 1),
        Err(Error::Misaligned {
            field: "ray_stride",
            ..
        })
    ));
    intersector.set_ray_stride(0);
    intersector
        .set_intersection_data_type(intersection_data_type::DISTANCE_PRIMITIVE_INDEX_COORDINATES);
    assert_eq!(
        encode(intersection_type::NEAREST, 0, 0, 1),
        Err(Error::BufferTooSmall {
            required: 16,
            length: 8,
        })
    );
    intersector.set_intersection_data_type(99);
    assert_eq!(
        intersector.intersection_data_type(),
        intersection_data_type::DISTANCE_PRIMITIVE_INDEX_COORDINATES
    );
    intersector.set_intersection_data_type(intersection_data_type::DISTANCE);
    encode(intersection_type::ANY, 0, 4, 1).expect("distance-only hits fit at offset 4");
    encode(intersection_type::NEAREST, 0, 0, 0).expect("no rays");
    command_buffer.commit().expect("commit");
    command_buffer.wait_until_completed().expect("completed");
}
