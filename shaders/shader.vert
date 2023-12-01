#version 460

layout( push_constant ) uniform CameraData{
    mat4 view;
    mat4 proj;
} camera;


layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inVelocity;

layout(location = 0) out vec2 outPosition;
layout(location = 1) out vec2 outVelocity;


void main() {
    gl_PointSize = 1;
    gl_Position = camera.proj * camera.view * vec4(inPosition, 0.0, 1.0);
    outPosition = inPosition;
    outVelocity = inVelocity;
}