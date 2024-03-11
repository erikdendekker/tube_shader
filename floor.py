import numpy as np

from OpenGL.GL import *

from renderable import Renderable
from opengl_utilities import create_opengl_program


vertex_shader = '''
#version 410 core

layout(location = 0) in vec2 vec;

out vec2 fpos;

uniform mat4 mvp;


void main()
{
    // Make 4D vertex from 2D value
    fpos = vec;
    vec4 v = vec4(vec.x, 0.0, vec.y, 1.0);
    gl_Position = mvp * v;
}
'''

fragment_shader = '''
#version 410 core

layout(location = 0) out vec4 fragment_color;

in vec2 fpos;


void main()
{
    float axis_width = 0.10;
    float half_axis_width = 0.5 * axis_width;

    if (abs(fpos.x) < half_axis_width && abs(fpos.y) < half_axis_width)
    {
        fragment_color = vec4(0.0, 1.0, 0.0, 1.0);
    }
    else if (fpos.x > 0 && fpos.x < 1 && abs(fpos.y) < half_axis_width)
    {
        fragment_color = vec4(1.0, 0.0, 0.0, 1.0);
    }
    else if (fpos.y > 0 && fpos.y < 1 && abs(fpos.x) < half_axis_width)
    {
        fragment_color = vec4(0.0, 0.0, 1.0, 1.0);
    }
    else if (mod(floor(fpos.x) + floor(fpos.y), 2) == 0)
    {
        fragment_color = vec4(0.6, 0.6, 0.6, 1.0);
    }
    else
    {
        fragment_color = vec4(0.4, 0.4, 0.4, 1.0);
    }

//    if (!gl_FrontFacing)
//    {
//        fragment_color = vec4(1.0, 0.0, 0.0, 1.0);
//    }
}
'''


class Floor(Renderable):

    def __init__(self, h_size: float, v_size: float):

        (self._shaders, self._shader_program) = create_opengl_program(vertex_shader=vertex_shader,
                                                                      fragment_shader=fragment_shader)

        self._mvp_location = glGetUniformLocation(self._shader_program, 'mvp')

        vertex_data = np.array([
            (-0.5 * h_size, -0.5 * v_size),
            (-0.5 * h_size, +0.5 * v_size),
            (+0.5 * h_size, -0.5 * v_size),
            (+0.5 * h_size, +0.5 * v_size)
        ], dtype=np.float32)

        # Make Vertex Buffer Object (VBO)
        self._vbo = glGenBuffers(1)

        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)

        # Create a vertex array object (VAO)
        # If a GL_ARRAY_BUFFER is bound, it will be associated with the VAO

        self._vao = glGenVertexArrays(1)
        glBindVertexArray(self._vao)

        # Defines the attribute with index 0 in the current VAO

        attribute_index = 0
        glVertexAttribPointer(attribute_index, 2, GL_FLOAT, GL_FALSE, 0, None)

        # Enable attribute with location 0
        glEnableVertexAttribArray(attribute_index)

        # Unbind VAO
        glBindVertexArray(0)

        # Unbind VBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def close(self):

        if self._vao is not None:
            glDeleteVertexArrays(1, (self._vao,))
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

        glDisable(GL_CULL_FACE)

        glBindVertexArray(self._vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 4)
