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
    float pad0;
    float pad1;           
} settings;

layout(location = 0) in vec3 inPosition;

vec4 transformScreenSpace(vec3 v){
    return vec4((settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) * (v.x / settings.DOMAIN_WIDTH) * 2.0 - (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT), (v.y / settings.DOMAIN_HEIGHT) * 2.0 , (v.z / settings.DOMAIN_HEIGHT) * 2.0 - 1.0, 1.0);
}

void main() {
    gl_PointSize = 1;
    gl_Position = ubo.proj * ubo.view * ubo.model * transformScreenSpace(inPosition);
}