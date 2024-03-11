""" Construct geometric objects as lists of triangles """
import numpy as np


def normalize(v) -> np.ndarray:
    """ Normalize vector """
    v = np.asarray(v)
    return v / np.linalg.norm(v)


def _make_unit_sphere_triangles_recursive(triangle, recursion_level: int) -> list:
    """ Perform recursive triangle subdivision and normalization to unit length """
    if recursion_level == 0:
        return [triangle]

    (v1, v2, v3) = triangle

    v12 = normalize(v1 + v2)
    v13 = normalize(v1 + v3)
    v23 = normalize(v2 + v3)

    triangles = []

    triangles.extend(_make_unit_sphere_triangles_recursive((v1, v12, v13), recursion_level - 1))
    triangles.extend(_make_unit_sphere_triangles_recursive((v12, v2, v23), recursion_level - 1))
    triangles.extend(_make_unit_sphere_triangles_recursive((v12, v23, v13), recursion_level - 1))
    triangles.extend(_make_unit_sphere_triangles_recursive((v13, v23, v3), recursion_level - 1))

    return triangles


def make_unit_sphere_triangles_tetrahedron(recursion_level: int):
    """ Make unit sphere by subdividing a tetrahedron """

    v1 = normalize((-1.0, -1.0, -1.0))
    v2 = normalize((+1.0, +1.0, -1.0))
    v3 = normalize((+1.0, -1.0, +1.0))
    v4 = normalize((-1.0, +1.0, +1.0))

    t1 = (v1, v2, v3)
    t2 = (v1, v4, v2)
    t3 = (v1, v3, v4)
    t4 = (v2, v4, v3)

    triangles = []

    triangles.extend(_make_unit_sphere_triangles_recursive(t1, recursion_level))
    triangles.extend(_make_unit_sphere_triangles_recursive(t2, recursion_level))
    triangles.extend(_make_unit_sphere_triangles_recursive(t3, recursion_level))
    triangles.extend(_make_unit_sphere_triangles_recursive(t4, recursion_level))

    return triangles


def make_unit_sphere_triangles(recursion_level: int):
    """ Make unit sphere by subdividing a dodecahedron """

    # Note: distance of center of each face to the origin is  0.7946544722917661,
    #   or sqrt((5+2*sqrt(5))/15)

    q = np.sqrt(5)
    r = np.sqrt((5 - q)/10)
    s = np.sqrt((5 + q)/10)

    vertices = np.array([
        (0, 0, -1),
        (0, 0, 1),
        (-2/q, 0, -1/q),
        (+2/q, 0, +1/q),
        ((+5 + q)/10, -r, -1/q),
        ((+5 + q)/10, +r, -1/q),
        ((-5 - q)/10, -r, +1/q),
        ((-5 - q)/10, +r, +1/q),
        ((-5 + q)/10, -s, -1/q),
        ((-5 + q)/10, +s, -1/q),
        ((+5 - q)/10, -s,  1/q),
        ((+5 - q)/10, +s,  1/q)
    ])

    face_definitions = [
        (1, 11, 7), (1, 7, 6), (1, 6, 10), (1, 10, 3),
        (1, 3, 11), (4, 8, 0), (5, 4, 0), (9, 5, 0),
        (2, 9, 0), (8, 2, 0), (11, 9, 7), (7, 2, 6),
        (6, 8, 10), (10, 4, 3), (3, 5, 11), (4, 10, 8),
        (5, 3, 4), (9, 11, 5), (2, 7, 9), (8, 6, 2)
    ]

    triangles = []

    for (v1_idx, v2_idx, v3_idx) in face_definitions:
        toplevel_triangle = (vertices[v1_idx], vertices[v2_idx], vertices[v3_idx])
        triangles.extend(_make_unit_sphere_triangles_recursive(toplevel_triangle, recursion_level))

    return triangles


def make_cylinder_triangles(subdivision_count: int, capped: bool):
    """ Make triangle mesh of a unit cylinder """

    z_lo = -0.5
    z_hi = +0.5

    triangles = []

    for i in range(subdivision_count):
        a0 = (i + 0) / subdivision_count * 2.0 * np.pi
        a1 = (i + 1) / subdivision_count * 2.0 * np.pi

        x0 = np.cos(a0)
        y0 = np.sin(a0)
        x1 = np.cos(a1)
        y1 = np.sin(a1)

        triangles.append(((x0, y0, z_lo), (x1, y1, z_lo), (x0, y0, z_hi)))
        triangles.append(((x1, y1, z_lo), (x1, y1, z_hi), (x0, y0, z_hi)))

    if capped:
        for i in range(subdivision_count):
            a0 = (i + 0) / subdivision_count * 2.0 * np.pi
            a1 = (i + 1) / subdivision_count * 2.0 * np.pi

            x0 = np.cos(a0)
            y0 = np.sin(a0)
            x1 = np.cos(a1)
            y1 = np.sin(a1)

            triangles.append(((0, 0, z_hi), (x0, y0, z_hi), (x1, y1, z_hi)))
            triangles.append(((0, 0, z_lo), (x1, y1, z_lo), (x0, y0, z_lo)))

    return triangles


def make_cylinder_placement_transform(p1, p2, diameter) -> np.ndarray:
    """ Create cylinder placement transformation matrix """
    # Assume unit cylinder with r=1 and extending from z= -0.5 to +0.5, scale its diameter and place it such that it extends from p1 to p2
    p1 = np.asarray(p1)
    p2 = np.asarray(p2)
    p_vector = p2 - p1

    # NOTE: Contains incorrect GUI warnings because np.cross() triggers a code reachability analysis bug in PyCharm!
    u = np.array((0, 0, 1))
    v = normalize(p_vector)
    rotation_angle = np.arccos(np.dot(u, v))
    rotation_vector = np.cross(u, v)

    if np.linalg.norm(rotation_vector) == 0:  # u is precisely parallel to v
       if rotation_angle > 0:
           orientation_matrix = scale(+1.0)   # Identity matrix
       else:
           orientation_matrix = scale(-1.0)
    else:
       orientation_matrix = rotate(rotation_vector, rotation_angle)

    # Apply a translation, scaling, rotation, and translation
    placement_matrix = translate(p1)                                         @ \
                       orientation_matrix                                    @ \
                       scale((diameter, diameter, np.linalg.norm(p_vector))) @ \
                       translate((0, 0, 0.5))

    return placement_matrix
