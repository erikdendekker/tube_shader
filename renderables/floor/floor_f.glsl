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
