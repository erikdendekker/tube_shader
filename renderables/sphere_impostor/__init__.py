import os

from PIL import Image

import numpy as np

from OpenGL.GL import *

from matrices import apply_transform_to_vertices
from renderables.renderable import Renderable
from renderables.opengl_utilities import create_opengl_program
from renderables.geometry import make_unit_sphere_triangles


class RenderableSphereImpostor(Renderable):

    def __init__(self, texture_path: str, m_xform=None):
        shader_source_path = os.path.join(os.path.dirname(__file__), "sphere_impostor")

        (self._shaders, self._shader_program) = create_opengl_program(shader_source_path)

        self._model_matrix_location      = glGetUniformLocation(self._shader_program, "model_matrix")
        self._view_matrix_location       = glGetUniformLocation(self._shader_program, "view_matrix")
        self._projection_matrix_location = glGetUniformLocation(self._shader_program, "projection_matrix")

        self._texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self._texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        with Image.open(texture_path) as im:
            image = np.array(im)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image.shape[1], image.shape[0], 0, GL_RGB, GL_UNSIGNED_BYTE, image)
        glGenerateMipmap(GL_TEXTURE_2D)

        triangles = make_unit_sphere_triangles(recursion_level=0)

        print("triangles:", len(triangles))

        triangle_vertices = np.array(triangles).reshape(-1, 3)

        triangle_vertices = np.multiply(triangle_vertices, 1.3)  # Oversize the impostor

        triangle_vertices = apply_transform_to_vertices(m_xform, triangle_vertices)

        print("triangle_vertices shape:", triangle_vertices.shape)

        vbo_dtype = np.dtype([
            ("a_vertex", np.float32, 3)
        ])

        vbo_data = np.empty(dtype=vbo_dtype, shape=len(triangle_vertices))

        vbo_data["a_vertex"] = triangle_vertices

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
        glVertexAttribPointer(attribute_index, 3, GL_FLOAT, GL_FALSE, 12, ctypes.c_void_p(0))
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

    def render(self, projection_matrix, view_matrix, model_matrix):

        glUseProgram(self._shader_program)

        glUniformMatrix4fv(self._model_matrix_location,      1, GL_TRUE, model_matrix     .astype(np.float32))
        glUniformMatrix4fv(self._view_matrix_location,       1, GL_TRUE, view_matrix      .astype(np.float32))
        glUniformMatrix4fv(self._projection_matrix_location, 1, GL_TRUE, projection_matrix.astype(np.float32))

        glBindTexture(GL_TEXTURE_2D, self._texture)
        glBindVertexArray(self._vao)
        glEnable(GL_CULL_FACE)
        glDrawArrays(GL_TRIANGLES, 0, self._num_points)
