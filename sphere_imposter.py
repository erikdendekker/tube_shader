from PIL import Image

import numpy as np

from OpenGL.GL import *

from matrices import apply_transform_to_vertices
from geometry import make_unit_sphere_triangles
from opengl_utilities import create_opengl_program
from renderable import Renderable


vertex_shader = '''
#version 410 core

layout(location = 0) in vec3 a_vertex;

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;

uniform mat4 projection_view_model_matrix;
uniform mat4 view_model_matrix;

out VS_OUT {
    vec3 mv_coordinate;
} vs_out;


void main()
{
    // Make 4D vertex from 3D value
    vec4 v = vec4(a_vertex, 1.0);

    vs_out.mv_coordinate = (view_model_matrix * v).xyz;

    gl_Position = projection_view_model_matrix * v;
}
'''

fragment_shader = '''
#version 410 core

out vec4 fragment_color;

in VS_OUT {
    vec3 mv_coordinate;
} fs_in;

uniform mat4 model_matrix;
uniform mat4 view_matrix;
uniform mat4 projection_matrix;

uniform mat4 view_model_matrix;
uniform mat4 inverse_view_model_matrix;
uniform mat4 transposed_inverse_view_matrix;
uniform mat4 transposed_inverse_view_model_matrix;
uniform mat4 projection_view_model_matrix;
uniform mat4 transposed_inverse_projection_view_model_matrix;

uniform sampler2D my_texture;

const float pi = 4 * atan(1);
const float nan = 0.0 / 0.0;


float intersect_unit_sphere(vec3 origin, vec3 direction)
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


// Phong constants
const float ia = 0.2;
const float id1 = 0.6;
const float is1 = 1.02;

const float phong_alpha = 20.0;


void main()
{
    // We receive model and modelview coordinates from the vertex shader

    // "e" is the eye position in the "unit sphere" coordinate system
    vec3 e = (inverse_view_model_matrix * vec4(0, 0, 0, 1)).xyz;

    // "h" is the impostor hitpoint position in the "unit sphere" coordinate system
    vec3 h = (inverse_view_model_matrix * vec4(fs_in.mv_coordinate, 1)).xyz;

    // Solve:    ray[alpha] := e + alpha * (h - e)
    // Find the smallest real value alpha such that ray[alpha]) intersects the unit sphere

    vec3 eh = h - e; // eye-to-hitpoint vector

    float alpha = intersect_unit_sphere(e, eh);
    if (isnan(alpha))
    {
        discard;
    }

    // This is the point where the ray and the unit sphere intersect in the "unit sphere" coordinate system
    // It is normalized since it is on the unit sphere
    vec3 sphere_hit = e + alpha * eh;

    // Find texture coordinates
    float u = 0.5 + 0.5 * atan(sphere_hit.x, sphere_hit.z) / pi;
    float v = 0.5 - 0.5 * sphere_hit.y;

    vec3 k_material = texture(my_texture, vec2(u, v)).xyz;

    // Fix depth
    vec4 projection = projection_view_model_matrix * vec4(sphere_hit, 1);
    gl_FragDepth = 0.5 + 0.5 *  (projection.z / projection.w);

    // Do phong shading
    vec3 mv_surface = (view_model_matrix * vec4(sphere_hit, 1)).xyz;
    vec3 mv_surface_normal = normalize((transposed_inverse_view_model_matrix * vec4(sphere_hit, 1)).xyz);
    vec3 mv_viewer_direction = normalize(-mv_surface);

    vec3 m_lightsource1_direction = normalize(vec3(+1, 1, 1));
    vec3 mv_lightsource1_direction = normalize((transposed_inverse_view_matrix * vec4(m_lightsource1_direction, 0)).xyz);
    vec3 mv_lightsource1_reflection_direction = 2 * dot(mv_lightsource1_direction, mv_surface_normal) * mv_surface_normal - mv_lightsource1_direction;

//  vec3 m_lightsource2_direction = normalize(vec3(-1, 1, 1));
//  vec3 mv_lightsource2_direction = normalize((transposed_inverse_view_matrix * vec4(m_lightsource2_direction, 0)).xyz);
//  vec3 mv_lightsource2_reflection_direction = 2 * dot(mv_lightsource2_direction, mv_surface_normal) * mv_surface_normal - mv_lightsource2_direction;

    float contrib_d1 = max(0.0, dot(mv_lightsource1_direction, mv_surface_normal));
    float contrib_s1 = pow(max(0.0, dot(mv_lightsource1_reflection_direction, mv_viewer_direction)), phong_alpha);

//  float contrib_d2 = max(0.0, dot(mv_lightsource2_direction, mv_surface_normal));
//  float contrib_s2 = pow(max(0.0, dot(mv_lightsource2_reflection_direction, mv_viewer_direction)), alpha);

    vec3 phong_color = k_material * (ia + id1 * contrib_d1 + is1 * contrib_s1);
//  vec3 phong_color = k_material * (ia + id1 * contrib_d1 + is1 * contrib_s1 + id2 * contrib_d2 + is2 * contrib_s2);

    fragment_color = vec4(phong_color, 1.0);
}
'''


class SphereImpostor(Renderable):

    def __init__(self, texture_path: str, m_xform=None):
        (self._shaders, self._shader_program) = create_opengl_program(vertex_shader=vertex_shader,
                                                                      fragment_shader=fragment_shader)

        self._model_matrix_location                                    = glGetUniformLocation(self._shader_program,      'model_matrix')
        self._view_matrix_location                                     = glGetUniformLocation(self._shader_program,       'view_matrix')
        self._projection_matrix_location                               = glGetUniformLocation(self._shader_program, 'projection_matrix')

        self._view_model_matrix_location                               = glGetUniformLocation(self._shader_program,                               "view_model_matrix")
        self._projection_view_model_matrix_location                    = glGetUniformLocation(self._shader_program,                    "projection_view_model_matrix")
        self._inverse_view_model_matrix_location                       = glGetUniformLocation(self._shader_program,                       "inverse_view_model_matrix")
        self._transposed_inverse_view_matrix_location                  = glGetUniformLocation(self._shader_program,                  "transposed_inverse_view_matrix")
        self._transposed_inverse_view_model_matrix_location            = glGetUniformLocation(self._shader_program,            "transposed_inverse_view_model_matrix")
        self._transposed_inverse_projection_view_model_matrix_location = glGetUniformLocation(self._shader_program, "transposed_inverse_projection_view_model_matrix")

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

        print(f'triangles: {len(triangles)}')

        triangle_vertices = np.array(triangles).reshape(-1, 3)

        triangle_vertices = np.multiply(triangle_vertices, 1.3)  # Oversize the impostor

        triangle_vertices = apply_transform_to_vertices(m_xform, triangle_vertices)

        print(f'triangle_vertices shape: {triangle_vertices.shape}')

        vbo_dtype = np.dtype([
            ('a_vertex', np.float32, 3)
        ])

        vbo_data = np.empty(dtype=vbo_dtype, shape=len(triangle_vertices))
        vbo_data['a_vertex'] = triangle_vertices

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

        glUniformMatrix4fv(self._model_matrix_location,                                    1, GL_TRUE, model_matrix     .astype(np.float32))
        glUniformMatrix4fv(self._view_matrix_location,                                     1, GL_TRUE, view_matrix      .astype(np.float32))
        glUniformMatrix4fv(self._projection_matrix_location,                               1, GL_TRUE, projection_matrix.astype(np.float32))

        # Precompute
        glUniformMatrix4fv(self._view_model_matrix_location,                               1, GL_TRUE,                                  (view_matrix @ model_matrix)  .astype(np.float32))
        glUniformMatrix4fv(self._projection_view_model_matrix_location,                    1, GL_TRUE,              (projection_matrix @ view_matrix @ model_matrix)  .astype(np.float32))
        glUniformMatrix4fv(self._inverse_view_model_matrix_location,                       1, GL_TRUE, np.linalg.inv(                    view_matrix @ model_matrix)  .astype(np.float32))
        glUniformMatrix4fv(self._transposed_inverse_view_matrix_location,                  1, GL_TRUE, np.linalg.inv(                    view_matrix               ).T.astype(np.float32))
        glUniformMatrix4fv(self._transposed_inverse_view_model_matrix_location,            1, GL_TRUE, np.linalg.inv(                    view_matrix @ model_matrix).T.astype(np.float32))
        glUniformMatrix4fv(self._transposed_inverse_projection_view_model_matrix_location, 1, GL_TRUE, np.linalg.inv(projection_matrix @ view_matrix @ model_matrix).T.astype(np.float32))

        glBindTexture(GL_TEXTURE_2D, self._texture)
        glBindVertexArray(self._vao)
        glEnable(GL_CULL_FACE)
        glDrawArrays(GL_TRIANGLES, 0, self._num_points)
