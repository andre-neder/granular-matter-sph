#version 460

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

layout(location = 0) in vec3 inPosition;
// layout(location = 1) in vec3 inColor;
// layout(location = 2) in vec3 inTexCoord;

// layout(location = 0) out vec3 fragColor;
// layout(location = 0) out vec4 outVelocity;
layout(location = 1) out vec3 outPos;
//
void main() {
    gl_Position = vec4(inPosition, 1);
    outPos = inPosition;
    // gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    // fragTexCoord = inTexCoord;
    // outVelocity = ubo.proj * ubo.view * ubo.model * vec4((settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) * (inVelocity.x / settings.DOMAIN_WIDTH) * 2.0 - (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT), (inVelocity.y / settings.DOMAIN_HEIGHT) * 2.0 - 1.0, 0.0, 1.0);
}