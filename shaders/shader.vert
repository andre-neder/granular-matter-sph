#version 460

// #define DOMAIN_WIDTH (9.f)
// #define DOMAIN_HEIGHT (9.f)

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
    float A_LR; 
    float v_max;
    float pad0;
    float pad1; 
} settings;

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vUV;
layout (location = 3) in vec4 vColor;
layout (location = 4) in vec4 vJoint;
layout (location = 5) in vec4 vWeight;
layout (location = 6) in vec4 vTangent;

layout(location = 7) in vec3 inPosition;
layout(location = 8) in vec3 inVelocity;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec3 eye;
layout(location = 2) out vec3 outVelocity;
layout(location = 3) out vec3 outPosition;

vec4 transformScreenSpace(vec3 v){
    return vec4(
        (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) * (v.x / settings.DOMAIN_WIDTH) * 2.0 - (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) - 1.0, 
        (v.y / settings.DOMAIN_HEIGHT) * 2.0, 
        (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) * (v.z / settings.DOMAIN_WIDTH) * 2.0 - (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) - 1.0, 
        1.0
    );
}




void main() {
    gl_PointSize = 1;
    // vec3 scale = inVelocity; // LR
    vec3 scale = vec3(settings.r_LR); // LR
    // float scale = settings.r_LR / 7; // HR

    mat4 model = mat4(1.0);
    model[0] = vec4(scale.x,0,0,0);
    model[1] = vec4(0,scale.y,0,0);
    model[2] = vec4(0,0,scale.z,0);
    model[3] = vec4(inPosition,1.0);
    eye = ubo.view[3].xyz;
    outNormal = (ubo.proj * ubo.view * vec4(vNormal, 0)).xyz;
    gl_Position = ubo.proj * ubo.view * transformScreenSpace( (model * vec4(vPosition, 1.0)).xyz) ;

    outPosition = inPosition;
    outVelocity = inVelocity;
}