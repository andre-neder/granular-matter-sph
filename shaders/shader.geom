#version 460
layout (triangles) in;
// layout (triangle_strip, max_vertices = 3) out;
layout (line_strip, max_vertices = 2) out;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;


layout(location = 0) in vec3 inNormal[];
layout(location = 1) in vec3 inEye[];
layout(location = 2) in vec3 inVelocity[];
layout(location = 3) in vec3 inPosition[];

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 outEye;
layout(location = 2) out vec3 outVelocity;
layout(location = 3) out vec3 outPosition;

layout( push_constant ) uniform Settings{
    vec4 g; 

    float r_LR;         
    float h_LR; 
    float rho0; 
    float mass;

    float maxCompression;	
    float dt;	 
    float DOMAIN_WIDTH; 
    float DOMAIN_HEIGHT;  

    float sleepingSpeed;
    float h_HR;
    float theta;                               
    float rhoAir;                                 
    
    vec4 windDirection;      

    float dragCoefficient;                
    uint n_HR; 
    float pad0;
    float pad1;           
} settings;

void main() {    
    gl_Position = ubo.proj * ubo.view * vec4(inPosition[0], 1.0);
    EmitVertex();

    gl_Position = ubo.proj * ubo.view * vec4(inPosition[0] + normalize(inVelocity[0]) * 0.2, 1.0);
    EmitVertex();

    EndPrimitive();
} 