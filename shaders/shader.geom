#version 460
layout (points) in;
layout (points, max_vertices = 5) out;
// layout (triangle_strip, max_vertices = 3) out;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in float[] inRho;
layout(location = 1) in vec3[] inPosition;

layout(location = 0) out float outRho;

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

vec4 transformScreenSpace(vec3 v){
    return vec4((settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) * (v.x / settings.DOMAIN_WIDTH) * 2.0 - (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT), (v.y / settings.DOMAIN_HEIGHT) * 2.0 - 1.0, (v.z / settings.DOMAIN_HEIGHT) * 2.0 - 1.0, 1.0);
}

void main() {    
    outRho = inRho[0];
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    EndPrimitive();
} 