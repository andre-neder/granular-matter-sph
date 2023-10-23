#version 460

// #define DOMAIN_WIDTH (9.f)
// #define DOMAIN_HEIGHT (9.f)

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout( push_constant ) uniform Settings{
    vec2 G; 
    float particleRadius;                 
    float kernelRadius; 

    float rho0; 
    float mass;
    float stiffness;	  
    float dt;	 

    float DOMAIN_WIDTH; 
    float DOMAIN_HEIGHT;  
    float pad1;
    float pad2;

    float theta;       
    float sigma;                           
    float alpha;                             
    float beta;                              
    
    float C;                                
    float dragCoefficient;                
    float rhoAir;                             
    float pad3;
} settings;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in float inRho;
// layout(location = 1) in vec3 inColor;
// layout(location = 2) in vec2 inTexCoord;

// layout(location = 0) out vec3 fragColor;
// layout(location = 0) out vec4 outVelocity;
layout(location = 0) out float outRho;
//
void main() {
    gl_PointSize = 1;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4((settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) * (inPosition.x / settings.DOMAIN_WIDTH) * 2.0 - (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT), (inPosition.y / settings.DOMAIN_HEIGHT) * 2.0 - 1.0, 0.0, 1.0);
    // gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    outRho = inRho;
    // fragTexCoord = inTexCoord;
    // outVelocity = ubo.proj * ubo.view * ubo.model * vec4((settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) * (inVelocity.x / settings.DOMAIN_WIDTH) * 2.0 - (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT), (inVelocity.y / settings.DOMAIN_HEIGHT) * 2.0 - 1.0, 0.0, 1.0);
}