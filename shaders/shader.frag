#version 460

layout( push_constant ) uniform CameraData{
    mat4 view;
    mat4 proj;
} camera;

layout(location = 0) out vec4 outColor;
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inVelocity;

void main() {
    outColor = vec4(246.f / 255.f,215.f / 255.f,176.f / 255.f, 1.0);
    // vec2 vel = (inPosition + inVelocity * settings.dt) - (inPosition);
    // if(length(vel) > settings.sleepingSpeed){
    //     outColor = vec4(0, 1, 0, 1);
    // }
    // else{
    //      outColor = vec4(1, 0, 0, 1);
    // }
    // outColor = vec4((inRho / settings.rho0 - 1) * 100, 1.0 - (inRho / settings.rho0 - 1) * 100, 0, 1);
    // outColor = vec4(inPad0, 1, 1, 1);
}