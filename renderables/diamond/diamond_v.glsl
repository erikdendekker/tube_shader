#version 410 core

layout (location = 0) in vec3 a_vertex;
layout (location = 1) in vec3 a_normal;
layout (location = 2) in ivec3 a_lattice_position;
layout (location = 3) in ivec3 a_lattice_delta;

uniform mat4 model_view_projection_matrix;
uniform mat4 model_view_matrix;
uniform mat4 transposed_inverse_model_view_matrix;

uniform uint cut_mode;
uniform uint unit_cells_per_dimension;
uniform float crystal_side_length;
uniform uint color_mode;

out VS_OUT {
    vec3 mv_surface;
    vec3 mv_normal;
    vec3 color;
} vs_out;

vec3 vec4_to_vec3(vec4 v)
{
    return v.xyz / v.w;
}

const float UNIT_CELL_SIZE = 4.0;

const vec3 cut100_normal = normalize(vec3(1.0, 0.0, 0.0));
const vec3 cut110_normal = normalize(vec3(1.0, 1.0, 0.0));
const vec3 cut111_normal = normalize(vec3(1.0, 1.0, 1.0));

const float positive_infinity = +1e30; // Certainly outside of crystal
const float negative_infinity = -1e30; // Certainly inside of crystal

const float cut_surface_threshold = 1e-3;


float crystal_lattice_surface_cut_distance(vec3 pos)
{
    // Positive : outside crystal
    // Negative : inside crystal

    bool candidate = max(abs(pos.x), max(abs(pos.y), abs(pos.z))) <= 0.5 * crystal_side_length;
    if (!candidate)
    {
        return positive_infinity; // Outside of crystal
    }

    switch (cut_mode)
    {
        case 1: return dot(cut100_normal, pos);
        case 2: return dot(cut110_normal, pos);
        case 3: return dot(cut111_normal, pos);
    }

    return negative_infinity; // Inside of crystal
}


void main()
{
    uint iz = gl_InstanceID;
    uint ix = iz % unit_cells_per_dimension; iz /= unit_cells_per_dimension;
    uint iy = iz % unit_cells_per_dimension; iz /= unit_cells_per_dimension;

    uvec3 unit_cell_index_vector = uvec3(ix, iy, iz);

    vec3 unit_cell_displacement_vector = UNIT_CELL_SIZE * (unit_cell_index_vector - 0.5 * unit_cells_per_dimension);

    // Do we want to render this triangle?

    vec3 lattice_position = unit_cell_displacement_vector + a_lattice_position;

    vec3 vertex_position = unit_cell_displacement_vector + a_vertex;

    float carbon_cut_distance = crystal_lattice_surface_cut_distance(lattice_position);

    bool render_flag = true;
    if (carbon_cut_distance > cut_surface_threshold)
    {
        render_flag = false;
    }
    else if (crystal_lattice_surface_cut_distance(lattice_position + a_lattice_delta) > cut_surface_threshold)
    {
        render_flag = false;
    }

    if (!render_flag)
    {
        // Emit a zero triangle, which will be discarded
        gl_Position = vec4(0.0, 0.0, 0.0, 1.0);
    }
    else
    {
        gl_Position = model_view_projection_matrix * vec4(vertex_position, 1.0);
        vs_out.mv_surface = vec4_to_vec3(model_view_matrix * vec4(vertex_position, 1.0));
        vs_out.mv_normal = normalize((transposed_inverse_model_view_matrix * vec4(a_normal, 0.0)).xyz);

        switch(color_mode)
        {
            case 0:
            {
                if (a_lattice_delta.x == 0)
                {
                    // Carbon atom (sphere)
                    vs_out.color = 0.2 + 0.9 * a_lattice_position / 3;
                }
                else
                {
                    // Carbon-carbon bond (cylinder)
                    vs_out.color = vec3(1.0, 1.0, 1.0);
                }
                break;
            }
            case 1:
            {
                if (a_lattice_delta.x == 0)
                {
                    // Carbon atom (sphere)
                    switch (cut_mode)
                    {
                        case 1: vs_out.color = (carbon_cut_distance > -1.5) ? vec3(1.0, 0.6, 0.6) : vec3(1.0, 1.0, 1.0); break;
                        case 2: vs_out.color = (carbon_cut_distance > -1.0) ? vec3(0.6, 1.0, 0.6) : vec3(1.0, 1.0, 1.0); break;
                        case 3: vs_out.color = (carbon_cut_distance > -1.9) ? vec3(0.6, 0.6, 1.0) : vec3(1.0, 1.0, 1.0); break;
                        default : vs_out.color = vec3(1.0, 1.0, 1.0); break;
                    }
                }
                else
                {
                    // Carbon-carbon bond (cylinder)
                    vs_out.color = vec3(1.0, 1.0, 1.0);
                }
                break;
            }
        }
    }
}
