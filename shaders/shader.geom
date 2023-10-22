#version 460
layout (points) in;
layout (points, max_vertices = 1) out;
// layout (line_strip, max_vertices = 2) out;
layout(location = 0) in vec4 velocity[];

void main() {    
    // gl_Position = gl_in[0].gl_Position;// + vec4(-0.1, 0.0, 0.0, 0.0); 
    // EmitVertex();

    // gl_Position = gl_in[0].gl_Position + vec4( 0.1, 0.0, 0.0, 0.0);
    // EmitVertex();
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();

    // gl_Position = gl_in[0].gl_Position + normalize(velocity[0]) * 0.1;;
    // EmitVertex();
    EndPrimitive();
} 