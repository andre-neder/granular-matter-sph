#version 450

#define DOMAIN_WIDTH (9.f)
#define DOMAIN_HEIGHT (9.f)

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition;
// layout(location = 1) in vec3 inColor;
// layout(location = 2) in vec2 inTexCoord;

// layout(location = 0) out vec3 fragColor;
// layout(location = 1) out vec2 fragTexCoord;
//
void main() {
    gl_PointSize = 1;
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4((inPosition.x / DOMAIN_WIDTH) * 2.0 - 1.0, (inPosition.y / DOMAIN_HEIGHT) * 2.0 - 1.0, 0.0, 1.0);
    // gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0);
    // fragColor = inColor;
    // fragTexCoord = inTexCoord;
}