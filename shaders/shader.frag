#version 460

// layout(binding = 1) uniform sampler2D texSampler; 
// texture(texSampler, fragTexCoord).rgb

// layout(location = 0) in vec3 fragColor;
// layout(location = 1) in vec2 fragTexCoord;


layout( push_constant ) uniform Settings{
    vec2 g; 
    float particleRadius;                 
    float kernelRadius; 

    float rho0; 
    float mass;
    float stiffness;	  
    float dt;	 

    float DOMAIN_WIDTH; 
    float DOMAIN_HEIGHT;  
    float sleepingSpeed;
    bool upsamplingEnabled;

    float theta;       
    float sigma;                           
    float alpha;                             
    float beta;                              
    
    float C;                                
    float dragCoefficient;                
    float rhoAir;                             
    float pad3;
} settings;

layout(location = 0) out vec4 outColor;
layout(location = 0) in float inRho;

void main() {
    // outColor = vec4(246.f / 255.f,215.f / 255.f,176.f / 255.f, 1.0);

    outColor = vec4((inRho / settings.rho0 - 1) * 10, 1.0 - (inRho / settings.rho0 - 1) * 10, 0, 1);
    // outColor = vec4(inPad0, 1, 1, 1);
}