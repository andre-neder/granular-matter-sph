#version 460

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

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
    float scale_W;
    float scale_GradW;           
} settings;

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vUV;
layout (location = 3) in vec4 vColor;
layout (location = 4) in vec4 vJoint;
layout (location = 5) in vec4 vWeight;
layout (location = 6) in vec4 vTangent;

layout (location = 0) out vec3 outPosition;
layout (location = 1) out vec3 outNormal;
layout (location = 2) out vec2 outUV;
void main() {
    gl_PointSize = 1;
    outUV = vUV;
    outPosition = (ubo.model * vec4(vPosition, 1.0)).xyz;
    outNormal = (transpose(inverse(ubo.model)) * vec4(vNormal, 0.0)).xyz;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(vPosition, 1.0);
}