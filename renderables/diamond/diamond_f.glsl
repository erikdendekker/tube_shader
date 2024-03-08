#version 410 core

uniform mat4 transposed_inverse_view_matrix;

layout(location = 0) out vec4 fragment_color;

in VS_OUT {
    vec3 mv_surface;
    vec3 mv_normal;
    vec3 color;
} fs_in;

// Intensities
const float ia = 0.2;
const float id1 = 0.6;
const float is1 = 1.02;
//const float id2 = 0.5;
//const float is2 = 0.5;

const float alpha = 100.0;


void main()
{

    vec3 k_material = fs_in.color;

    // NOTE: We do our geometric calculations in the "MV" coordinate system

    vec3 mv_eye = vec3(0, 0, 0);
    vec3 mv_surface = fs_in.mv_surface;
    vec3 mv_surface_normal = normalize(fs_in.mv_normal);
    vec3 mv_viewer_direction = normalize(mv_eye - mv_surface);

    vec3 m_lightsource1_direction = normalize(vec3(+1, 1, 1));
    vec3 mv_lightsource1_direction = normalize((transposed_inverse_view_matrix * vec4(m_lightsource1_direction, 0)).xyz);
    vec3 mv_lightsource1_reflection_direction = 2 * dot(mv_lightsource1_direction, mv_surface_normal) * mv_surface_normal - mv_lightsource1_direction;

    vec3 m_lightsource2_direction = normalize(vec3(-1, 1, 1));
    vec3 mv_lightsource2_direction = normalize((transposed_inverse_view_matrix * vec4(m_lightsource2_direction, 0)).xyz);
    vec3 mv_lightsource2_reflection_direction = 2 * dot(mv_lightsource2_direction, mv_surface_normal) * mv_surface_normal - mv_lightsource2_direction;

    float contrib_d1 = max(0.0, dot(mv_lightsource1_direction, mv_surface_normal));
    float contrib_s1 = pow(max(0.0, dot(mv_lightsource1_reflection_direction, mv_viewer_direction)), alpha);

    //float contrib_d2 = max(0.0, dot(mv_lightsource2_direction, mv_surface_normal));
    //float contrib_s2 = pow(max(0.0, dot(mv_lightsource2_reflection_direction, mv_viewer_direction)), alpha);

    vec3 phong_color = k_material * (ia + id1 * contrib_d1 + is1 * contrib_s1);
    //vec3 phong_color = k_material * (ia + id1 * contrib_d1 + is1 * contrib_s1 + id2 * contrib_d2 + is2 * contrib_s2);

    fragment_color = vec4(phong_color, 1.0);
}
