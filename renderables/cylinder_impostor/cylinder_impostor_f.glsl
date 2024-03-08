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


float squared(float x)
{
    return x * x;
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
    float eh_xy_dot_eh_xy = dot(eh.xy, eh.xy);

    float discriminant = eh_xy_dot_eh_xy - squared(e.x * h.y - e.y * h.x);

    if (discriminant < 0)
    {
        fragment_color = vec4(1.0, 1.0, 0.0, 1.0);
        return;
        // The ray that hits the impostor doesn't hit the enclosed cylinder
        discard;
    }

    float e_xy_dot_e_xy = dot(e.xy, e.xy);
    float e_xy_dot_h_xy = dot(e.xy, h.xy);

    float alpha = (e_xy_dot_e_xy - e_xy_dot_h_xy - sqrt(discriminant)) / eh_xy_dot_eh_xy;

    // This is the point where the ray and the unit cylinder intersect in the "unit cylinder" coordinate system
    // Its xy coordinates are normalized since they are a point on the unit cylinder
    vec3 cylinder_hit = e + alpha * eh;

    if (abs(cylinder_hit.z) > 0.5)
    {
        // The cylinder is hit, but outside its z range [-0.5 .. +0.5]
        fragment_color = vec4(0.0, 1.0, 1.0, 1.0);
        discard;
    }

    float u = 0.5 + 0.5 * atan(cylinder_hit.x, cylinder_hit.y) / pi;
    float v = 0.5 + cylinder_hit.z;

    fragment_color = texture(my_texture, vec2(u, v));

    vec4 proj = projection_matrix * view_matrix * model_matrix * vec4(cylinder_hit, 1);
    float depth = (proj.z / proj.w);

    gl_FragDepth = 0.5 + 0.5 * depth;
}
