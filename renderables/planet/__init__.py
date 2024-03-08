import os

from PIL import Image

import numpy as np

from OpenGL.GL import *

from matrices import apply_transform_to_vertices
from renderables.renderable import Renderable
from renderables.opengl_utilities import create_opengl_program
from renderables.geometry import make_unit_sphere_triangles, make_unit_sphere_triangles_tetrahedron


class RenderablePlanet(Renderable):

    def __init__(self, texture_path : str, m_xform=None):

        shader_source_path = os.path.join(os.path.dirname(__file__), "planet")

        (self._shaders, self._shader_program) = create_opengl_program(shader_source_path)

        self._mvp_location = glGetUniformLocation(self._shader_program, "mvp")

        self._texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        with Image.open(texture_path) as im:
            image = np.array(im)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.shape[1], image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image)

        triangles = make_unit_sphere_triangles(recursion_level=5)

        print("triangles:", len(triangles))

        triangle_vertices = np.array(triangles).reshape(-1, 3)

        triangle_vertices = apply_transform_to_vertices(m_xform, triangle_vertices)

        print("triangle_vertices shape:", triangle_vertices.shape)

        vbo_dtype = np.dtype([
            ("a_vertex", np.float32, 3),
            ("a_texture_coordinate", np.float32, 2)
        ])

        vbo_data = np.empty(dtype=vbo_dtype, shape=len(triangle_vertices))

        vbo_data["a_vertex"] = triangle_vertices
        vbo_data["a_texture_coordinate"][:, 0] = 0.5 + 0.5 * np.arctan2(triangle_vertices[:, 0], triangle_vertices[:, 2]) / np.pi
        vbo_data["a_texture_coordinate"][:, 1] = 0.5 - 0.5 * triangle_vertices[:, 1]

        # Prevent texture coordinate wrap-around
        vbo_data["a_texture_coordinate"][1::3, 0] = vbo_data["a_texture_coordinate"][0::3, 0] + (vbo_data["a_texture_coordinate"][1::3, 0] - vbo_data["a_texture_coordinate"][0::3, 0] + 0.5) % 1.0 - 0.5
        vbo_data["a_texture_coordinate"][2::3, 0] = vbo_data["a_texture_coordinate"][0::3, 0] + (vbo_data["a_texture_coordinate"][2::3, 0] - vbo_data["a_texture_coordinate"][0::3, 0] + 0.5) % 1.0 - 0.5

        self._num_points = len(vbo_data)

        # Make Vertex Buffer Object (VBO)
        self._vbo = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vbo_data.nbytes, vbo_data, GL_STATIC_DRAW)

        # Create a vertex array object (VAO)
        # If a GL_ARRAY_BUFFER is bound, it will be associated with the VAO

        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)

        # Defines the attribute with index 0 in the current VAO

        attribute_index = 0  # 3D vertex coordinates
        glVertexAttribPointer(attribute_index, 3, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(0))
        glEnableVertexAttribArray(attribute_index)

        attribute_index = 1  # 2D texture coordinates
        glVertexAttribPointer(attribute_index, 2, GL_FLOAT, GL_FALSE, 20, ctypes.c_void_p(12))
        glEnableVertexAttribArray(attribute_index)

        # Unbind VAO
        glBindVertexArray(0)

        # Unbind VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def close(self):

        if self._vao is not None:
            glDeleteVertexArrays(1, (self._vao, ))
            self._vao = None

        if self._vbo is not None:
            glDeleteBuffers(1, (self._vbo, ))
            self._vbo = None

        if self._shader_program is not None:
            glDeleteProgram(self._shader_program)
            self._shader_program = None

        if self._shaders is not None:
            for shader in self._shaders:
                glDeleteShader(shader)
            self._shaders = None

    def render(self, m_projection, m_view, m_model):

        glUseProgram(self._shader_program)

        mvp = m_projection @ m_view @ m_model
        glUniformMatrix4fv(self._mvp_location, 1, GL_TRUE, mvp.astype(np.float32))

        glBindTexture(GL_TEXTURE_2D, self._texture)
        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLES, 0, self._num_points)
