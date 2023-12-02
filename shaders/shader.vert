#version 460

// #define DOMAIN_WIDTH (9.f)
// #define DOMAIN_HEIGHT (9.f)

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout( push_constant ) uniform Settings{
    vec2 g; 
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
    float sigma;                           
    float alpha;                             
    uint n_HR;                              
    
    float pad2;                                
    float dragCoefficient;                
    float rhoAir;                             
    float pad3;
} settings;

layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inVelocity;

layout(location = 0) out vec2 outPosition;
layout(location = 1) out vec2 outVelocity;

vec4 transformScreenSpace(vec2 v){
    return vec4((settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) * (v.x / settings.DOMAIN_WIDTH) * 2.0 - (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT), (v.y / settings.DOMAIN_HEIGHT) * 2.0 - 1.0, 0.0, 1.0);
}

void main() {
    gl_PointSize = 1;
    // gl_Position = vec4(inPosition, 0, 1);
    gl_Position = ubo.proj * ubo.view * ubo.model * transformScreenSpace(inPosition) ;
    outPosition = inPosition;
    outVelocity = inVelocity;
    // outRho = inRho;
}