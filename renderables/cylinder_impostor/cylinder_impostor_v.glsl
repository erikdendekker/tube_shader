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
