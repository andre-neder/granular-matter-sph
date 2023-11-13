#version 460

layout( push_constant ) uniform Settings{
    vec2 g; 
    float r_LR;                 
    float h_LR; 

    float rho0; 
    float mass;
    float stiffness;	  
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
// layout(location = 1) in vec3 inColor;
// layout(location = 2) in vec2 inTexCoord;

// layout(location = 0) out vec3 fragColor;
// layout(location = 0) out vec4 outVelocity;
layout(location = 1) out vec2 outPos;
//
void main() {
    gl_Position = vec4(inPosition, 0, 1);
    outPos = inPosition;
    // gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    // fragTexCoord = inTexCoord;
    // outVelocity = ubo.proj * ubo.view * ubo.model * vec4((settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) * (inVelocity.x / settings.DOMAIN_WIDTH) * 2.0 - (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT), (inVelocity.y / settings.DOMAIN_HEIGHT) * 2.0 - 1.0, 0.0, 1.0);
}