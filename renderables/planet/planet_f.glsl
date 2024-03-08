#version 410 core

out vec4 fragment_color;

in vec2 tex_coord;

uniform sampler2D my_texture;


void main()
{
    fragment_color = texture(my_texture, tex_coord);

    if (false && !gl_FrontFacing)
    {
        float monochrome = (0.2125 * fragment_color.r) + (0.7154 * fragment_color.g) + (0.0721 * fragment_color.b);
        fragment_color = vec4(monochrome, monochrome, monochrome, 1.0);
    }
}
