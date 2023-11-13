#version 460

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

void main() {
    gl_PointSize = 1;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4((settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) * (inPosition.x / settings.DOMAIN_WIDTH) * 2.0 - (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT), (inPosition.y / settings.DOMAIN_HEIGHT) * 2.0 - 1.0, 0.0, 1.0);
}