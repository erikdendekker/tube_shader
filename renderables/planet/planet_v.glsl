#version 410 core

layout(location = 0) in vec3 vertex;
layout(location = 1) in vec2 texture_coordinate;

uniform mat4 mvp;

out vec2 tex_coord;


void main()
{
    // Make 4D vertex from 3D value
    vec4 v = vec4(vertex, 1.0);
    gl_Position = mvp * v;
    tex_coord = texture_coordinate;
}
