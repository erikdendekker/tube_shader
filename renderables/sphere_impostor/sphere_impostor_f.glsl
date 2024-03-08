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


void main()
{
    // We receive model and modelview coordinates from the vertex shader

    mat4 inverse_model_view_matrix = inverse(view_matrix * model_matrix);

    // "e" is the eye position in the "unit sphere" coordinate system
    vec3 e = (inverse_model_view_matrix * vec4(0, 0, 0, 1)).xyz;

    // "h" is the impostor hitpoint position in the "unit sphere" coordinate system
    vec3 h = (inverse_model_view_matrix * vec4(fs_in.mv_coordinate, 1)).xyz;

    // Solve:    ray[alpha] := e + alpha * (h - e)
    // Find the smallest real value alpha such that ray[alpha]) intersects the unit sphere

    vec3 eh = h - e; // eye-to-hitpoint vector

    float eh_dot_eh = dot(eh, eh);
    vec3  e_cross_h = cross(e, h);
    float e_cross_h_dot_e_cross_h = dot(e_cross_h, e_cross_h);

    float discriminant = eh_dot_eh - e_cross_h_dot_e_cross_h;

    if (discriminant < 0)
    {
        discard; // The ray that hits the impostor doesn't hit the enclosed sphere
    }

    float e_dot_e = dot(e, e);
    float e_dot_h = dot(e, h);

    float alpha = (e_dot_e - e_dot_h - sqrt(discriminant)) / eh_dot_eh;

    // This is the point where the ray and the unit sphere intersect in the "unit sphere" coordinate system
    // It is normalized since it is on the unit sphere
    vec3 sphere_hit = e + alpha * eh;

    // Find texture coordinates
    float u = 0.5 + 0.5 * atan(sphere_hit.x, sphere_hit.z) / pi;
    float v = 0.5 - 0.5 * sphere_hit.y;

    fragment_color = texture(my_texture, vec2(u, v));

    vec4 projection = projection_matrix * view_matrix * model_matrix * vec4(sphere_hit, 1);

    gl_FragDepth = 0.5 + 0.5 *  (projection.z / projection.w);
}
