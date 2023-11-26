#version 460

layout(location = 0) in vec2 inPosition;
layout(location = 1) out vec2 outPos;

void main() {
    gl_Position = vec4(inPosition, 0, 1);
    outPos = inPosition;
}