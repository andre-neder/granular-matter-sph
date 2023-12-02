#version 460

// layout(binding = 1) uniform sampler3D texSampler; 
// texture(texSampler, fragTexCoord).rgb

// layout(location = 0) in vec3 fragColor;
// layout(location = 1) in vec3 fragTexCoord;


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

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inVelocity;

void main() {
    outColor = vec4(246.f / 255.f,215.f / 255.f,176.f / 255.f, 1.0);
    // vec3 vel = (inPosition + inVelocity * settings.dt) - (inPosition);
    // if(length(vel) > settings.sleepingSpeed){
    //     outColor = vec4(0, 1, 0, 1);
    // }
    // else{
    //      outColor = vec4(1, 0, 0, 1);
    // }
    // outColor = vec4((inRho / settings.rho0 - 1) * 100, 1.0 - (inRho / settings.rho0 - 1) * 100, 0, 1);
    // outColor = vec4(inPad0, 1, 1, 1);
}