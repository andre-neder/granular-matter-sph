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
    float scale_W;
    float scale_GradW;
    float A_LR; 
    float v_max;
    float pad0;
    float pad1; 
} settings;

layout(location = 0) out vec4 outColor;

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec3 eye;
layout(location = 2) in vec3 inVelocity;
layout(location = 3) in vec3 inPosition;
layout(location = 9) in vec4 inColor;

vec3 lightDir = vec3(1.0, 1.0, 1.0);
void main() {
    vec4 sand = inColor; // vec4(inVelocity, 1.0); //vec4(vec3(length(inVelocity), 0.0, 1 - length(inVelocity)),1.0); //
    vec4 lightColor = vec4(1.0);
    vec4 ambient = vec4(0.3);

    float phi = max(dot(normalize(inNormal), normalize(eye)), 0.0);
    outColor = sand * ambient + sand * cos(1 - phi) * lightColor;
    
    // float diff = max(dot(normalize(inNormal), lightDir), 0.0);
    // outColor = (0.03 + diff) * inColor;
    // vec3 vel = (inPosition + inVelocity * settings.dt) - (inPosition);
    // if(length(vel) > settings.sleepingSpeed){
    //     outColor = vec4(0, 1, 0, 1);
    // }
    // else{
    //      outColor = vec4(1, 0, 0, 1);
    // }
    // outColor = vec4((inRho / settings.rho0 - 1) * 100, 1.0 - (inRho / settings.rho0 - 1) * 100, 0, 1);
    // outColor = vec4(0, 1, 1, 1);
}