""" Construct geometric objects as lists of triangles """
import numpy as np


def normalize(v) -> np.ndarray:
    """ Normalize vector """
    vec = np.asarray(v)
    return vec / np.linalg.norm(vec)


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


def make_cylinder_triangles(subdivision_count: int, caps: bool) -> tuple[list, list]:

    zlo = -0.5
    zhi = +0.5

    normals = []
    triangles = []
    for i in range(subdivision_count):
        a0 = (i + 0) / subdivision_count * 2.0 * np.pi
        a1 = (i + 1) / subdivision_count * 2.0 * np.pi

        x0 = np.cos(a0)
        y0 = np.sin(a0)
        x1 = np.cos(a1)
        y1 = np.sin(a1)

        triangle = ((x0, y0, zlo), (x1, y1, zlo), (x0, y0, zhi))
        normal = ((x0, y0, 0), (x1, y1, 0), (x0, y0, 0))
        triangles.append(triangle)
        normals.append(normal)

        triangle = ((x1, y1, zlo), (x1, y1, zhi), (x0, y0, zhi))
        normal = ((x1, y1, 0), (x1, y1, 0), (x0, y0, 0))
        triangles.append(triangle)
        normals.append(normal)

    if caps:
        for i in range(subdivision_count):
            a0 = (i + 0) / subdivision_count * 2.0 * np.pi
            a1 = (i + 1) / subdivision_count * 2.0 * np.pi

            x0 = np.cos(a0)
            y0 = np.sin(a0)
            x1 = np.cos(a1)
            y1 = np.sin(a1)

            triangle = ((0, 0, zhi), (x0, y0, zhi), (x1, y1, zhi))
            normal = ((0, 0, 1), (0, 0, 1), (0, 0, 1))
            triangles.append(triangle)
            normals.append(normal)
            triangle = ((0, 0, zlo), (x1, y1, zlo), (x0, y0, zlo))
            normal = ((0, 0, -1), (0, 0, -1), (0, 0, -1))
            triangles.append(triangle)
            normals.append(normal)

    return triangles, normals
