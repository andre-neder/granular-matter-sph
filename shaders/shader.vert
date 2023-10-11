#version 450

// #define DOMAIN_WIDTH (9.f)
// #define DOMAIN_HEIGHT (9.f)

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(binding = 1) uniform SPHSettings {
    vec2 G;       
    float rhoRest;
    float kernelRadius;		   

    float mass;		 
    float stiffness;
    float dt;	  
    float DOMAIN_WIDTH;

    float DOMAIN_HEIGHT;
    float pad0, pad1, pad2;

} settings;

layout(location = 0) in vec3 inPosition;
layout(location = 1) in float rho;
// layout(location = 1) in vec3 inColor;
// layout(location = 2) in vec2 inTexCoord;

layout(location = 0) out vec3 fragColor;
// layout(location = 1) out vec2 fragTexCoord;
//
void main() {
    gl_PointSize = 1;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4((inPosition.x / settings.DOMAIN_WIDTH) * 2.0 - 1.0, (inPosition.y / settings.DOMAIN_HEIGHT) * 2.0 - 1.0, 0.0, 1.0);
    // gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    fragColor = vec3(0.f, rho / settings.rhoRest, 1.f);
    // fragTexCoord = inTexCoord;
}