from PIL import Image

import numpy as np

from OpenGL.GL import *

from matrices import apply_transform_to_vertices
from renderable import Renderable
from opengl_utilities import create_opengl_program
from geometry import make_cylinder_triangles


vertex_shader = \
'''
#version 410 core

layout(location = 0) in vec3 a_vertex;

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;

out VS_OUT {
    vec3 m_coordinate;
    vec3 mv_coordinate;
} vs_out;


void main()
{
    // Make 4D vertex from 3D value
    vec4 v = vec4(a_vertex, 1.0);
    mat4 mvp = (projection_matrix * view_matrix * model_matrix);

    vs_out.m_coordinate = (model_matrix * v).xyz;
    vs_out.mv_coordinate = (view_matrix * model_matrix * v).xyz;

    gl_Position = mvp * v;
}
'''


fragment_shader = \
'''
#version 410 core

out vec4 fragment_color;

in VS_OUT {
    vec3 m_coordinate;
    vec3 mv_coordinate;
} fs_in;

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;

uniform sampler2D my_texture;

const float pi = 4 * atan(1);
const float nan = 0.0 / 0.0;


float intersect_unit_cylinder(vec2 origin, vec2 direction)
{
    // See: https://en.wikipedia.org/wiki/Line–sphere_intersection
    // Find smallest real alpha such that: origin + alpha * direction is on the unit circle
    float oo = dot(origin, origin);
    float uo = dot(direction, origin);
    float uu = dot(direction, direction);
    float discriminant = uo*uo - uu * (oo - 1);

    // Early abort if a solution does not exist (Check can be omitted, but it is adventageous to keep it for improved performance)
    if (discriminant < 0)
    {
        return nan;
    }

    return (-uo - sqrt(discriminant)) / uu;
}


void main()
{
    // We receive model and modelview coordinates from the vertex shader

    // Get the eye coordinate in "original object vertex" coordinates:

    mat4 inverse_model_view_matrix = inverse(view_matrix * model_matrix);

    // "e" is the eye position in the "unit cylinder" coordinate system
    vec3 e = (inverse_model_view_matrix * vec4(0, 0, 0, 1)).xyz;

    // "h" is the impostor hitpoint position in the "unit cylinder" coordinate system
    vec3 h = (inverse_model_view_matrix * vec4(fs_in.mv_coordinate, 1)).xyz;

    // Solve:
    //   ray[alpha] := e + alpha * (h - e)
    // Find the smallest real value alpha such that ray[alpha]) intersects the unit cylinder

    vec3 eh = h - e;

    float alpha = intersect_unit_cylinder(e.xy, eh.xy);

     if (isnan(alpha))
    {
//      fragment_color = vec4(1.0, 1.0, 0.0, 1.0);
//      return;
        discard;
    }

    // This is the point where the ray and the unit cylinder intersect in the "unit cylinder" coordinate system
    // Its xy coordinates are normalized since they are a point on the unit cylinder
    vec3 cylinder_hit = e + alpha * eh;

    if (abs(cylinder_hit.z) > 0.5)
    {
        // The cylinder is hit, but outside its z range [-0.5 .. +0.5]
//      fragment_color = vec4(0.0, 1.0, 1.0, 1.0);
//      return;
        discard;
    }

    float u = 0.5 + 0.5 * atan(cylinder_hit.x, cylinder_hit.y) / pi;
    float v = 0.5 + cylinder_hit.z;

    fragment_color = texture(my_texture, vec2(u, v));

    vec4 proj = projection_matrix * view_matrix * model_matrix * vec4(cylinder_hit, 1);
    float depth = (proj.z / proj.w);

    gl_FragDepth = 0.5 + 0.5 * depth;
}
'''


class CylinderImpostor(Renderable):

    def __init__(self, texture_path : str, m_xform=None):

        (self._shaders, self._shader_program) = create_opengl_program(vertex_shader=vertex_shader,
                                                                      fragment_shader=fragment_shader)

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

        triangles = make_cylinder_triangles(subdivision_count=6, caps=True)

        print("triangles:", len(triangles))

        triangle_vertices = np.array(triangles).reshape(-1, 3)

        triangle_vertices = np.multiply(triangle_vertices, (1.25, 1.25, 1.05))  # Oversize the impostor

        triangle_vertices = apply_transform_to_vertices(m_xform, triangle_vertices)

        print("triangle_vertices shape:", triangle_vertices.shape)

        vbo_dtype = np.dtype([
            ("a_vertex", np.float32, 3)
        ])

        vbo_data = np.empty(dtype=vbo_dtype, shape=len(triangle_vertices))

        vbo_data["a_vertex"] = triangle_vertices  # Oversize the impostor

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
