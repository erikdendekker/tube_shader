""" Construct and apply OpenGL transformation matrices """
import numpy as np


def translate(translation_vector, dtype=None) -> np.ndarray:
    """ Create 4x4 translation matrix """

    if dtype is None:
        dtype = np.float64

    t = np.asarray(translation_vector, dtype=dtype)

    if t.shape != (3, ):
        raise ValueError('Bad translation_vector argument')

    return np.array([
        [1, 0, 0, t[0]],
        [0, 1, 0, t[1]],
        [0, 0, 1, t[2]],
        [0, 0, 0,  1]
    ], dtype=dtype)


def scale(scale_coefficients, dtype=None) -> np.ndarray:
    """ Create 4x4 uniform or non-uniform scaling matrix

    The scale argument should either be a 1- or 3-element vector of scale coefficients.
    """

    if dtype is None:
        dtype = np.float64

    s = np.asarray(scale_coefficients, dtype=dtype)

    if s.ndim == 0:
        s = np.repeat(s, 3)

    if s.shape != (3, ):
        raise ValueError('Bad scale_coefficients argument')

    return np.array([
        [s[0], 0, 0, 0],
        [0, s[1], 0, 0],
        [0, 0, s[2], 0],
        [0, 0, 0, 1]
    ], dtype=dtype)


def rotate(rotation_axis, angle: float, dtype=None) -> np.ndarray:
    """ Create rotation matrix """

    if dtype is None:
        dtype = np.float64

    r = np.asarray(rotation_axis, dtype=dtype)

    if r.shape != (3, ):
        raise ValueError('Bad rotation_axis argument')

    r /= np.linalg.norm(r)

    ca = np.cos(angle)
    sa = np.sin(angle)

    cca = 1 - ca

    return np.array([
        [cca * r[0] * r[0] + ca, cca * r[1] * r[0] - r[2] * sa, cca * r[2] * r[0] + r[1] * sa, 0],
        [cca * r[0] * r[1] + r[2] * sa, cca * r[1] * r[1] + ca, cca * r[2] * r[1] - r[0] * sa, 0],
        [cca * r[0] * r[2] - r[1] * sa, cca * r[1] * r[2] + r[0] * sa, cca * r[2] * r[2] + ca, 0],
        [0, 0, 0, 1]
    ], dtype=dtype)


def frustum(left: float, right: float, bottom: float, top: float, near: float, far: float, dtype=None) -> np.ndarray:
    """ Create frustum projection matrix """

    if dtype is None:
        dtype = np.float64

    return np.array([
        [2 * near / (right - left), 0, (right + left) / (right - left), 0],
        [0, 2 * near / (top - bottom), (top + bottom) / (top - bottom), 0],
        [0, 0, (far + near) / (near - far), 2 * far * near / (near - far)],
        [0, 0, -1, 0]
    ], dtype=dtype)


def perspective_projection(framebuffer_width: int, framebuffer_height: int,
                           fov_degrees: float, near: float, far: float, dtype=None) -> np.ndarray:
    """ Create perspective projection matrix """

    if dtype is None:
        dtype = np.float64

    fov_radians = np.radians(fov_degrees)

    if framebuffer_width >= framebuffer_height:
        frustum_height = 2.0 * near * np.tan(0.5 * fov_radians)
        frustum_width = frustum_height * framebuffer_width / framebuffer_height
    else:
        frustum_width = 2.0 * near * np.tan(0.5 * fov_radians)
        frustum_height = frustum_width * framebuffer_height / framebuffer_width

    return frustum(
        -0.5 * frustum_width,  +0.5 * frustum_width,
        -0.5 * frustum_height, +0.5 * frustum_height,
        near, far, dtype=dtype)


def apply_transform_to_vertices(m_xform: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """ Apply transform to array of vertices """
    if m_xform is None:
        return vertices

    ok = m_xform.shape == (4, 4) and (vertices.ndim == 2) and (vertices.shape[1] == 3)
    if not ok:
        raise ValueError('Bad transform requested')

    # Append constant '1' column
    vertices = np.pad(vertices, pad_width=((0, 0), (0, 1)), constant_values=1.0)
    vertices = (m_xform @ vertices.T).T
    vertices = np.delete(vertices, -1, 1)

    return vertices


def apply_transform_to_normals(m_xform: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """ Apply transform to array of normal vectors """

    if m_xform is None:
        return normals

    ok = m_xform.shape == (4, 4) and (normals.ndim == 2) and (normals.shape[1] == 3)
    if not ok:
        raise ValueError('Bad transform requested')

    # Append constant '0' column
    normals = np.pad(normals, pad_width=((0, 0), (0, 1)), constant_values=0.0)

    normals = (np.linalg.inv(m_xform).T @ normals.T).T
    normals = np.delete(normals, -1, 1)

    # Normalize normals to unit length
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    if not np.all(np.abs(np.linalg.norm(normals, axis=1) - 1.0) < 1e-12):
        raise RuntimeError('normals are not unit length after normalization')

    return normals
