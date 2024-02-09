#include "camera.h"
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include "input.h"

using namespace gpu;

Camera::Camera(){
    
}

Camera::Camera(Type type, GLFWwindow* window, uint32_t width, uint32_t height, glm::vec3 eye, glm::vec3 center)
{
    m_window = window;
    _radius = glm::length(eye - center);
    _type = type;
    _center = center;
    _eye = eye;
    _forward = glm::normalize(_center - _eye);
    if(type == Type::eFirstPerson) {
        _angle = glm::vec2(atan2(_forward.x, _forward.z), asin(_forward.y) + 1.570796f);
    } else {
        glm::vec3 _eyeNorm = glm::normalize(_eye);
        _angle = glm::vec2(atan2(_eyeNorm.x, _eyeNorm.z), asin(-_eyeNorm.y) + 1.570796f);
    }
    _width = width;
    _height = height;
    _oldPos = glm::vec2(width / 2.0f, height / 2.0f);
    _newPos = glm::vec2(width / 2.0f, height / 2.0f);

    update(0);
}


void Camera::update(float dt){

    double delta_time = dt;
    float _xpos = gpu::InputManager::cursorPosition.x;
    float _ypos = gpu::InputManager::cursorPosition.y;

    Camera::_radius -= (float) (gpu::InputManager::scrollOffset.y * 0.1);
    Camera::_radius = glm::max((float)Camera::_radius, 0.000001f);

    if (gpu::InputManager::isMouseButtonDown(GLFW_MOUSE_BUTTON_LEFT)) {
        _newPos = glm::vec2(_xpos, _ypos);
        float xAngle = (_newPos.x - _oldPos.x) / _width * 2 * 3.141592f;
        float yAngle = (_newPos.y - _oldPos.y) / _height * 3.141592f;
        _angle -= glm::vec2(xAngle, yAngle);
        _angle.y = glm::max(glm::min(_angle.y, 3.141591f), 0.000001f);
    }
    _oldPos = glm::vec2(_xpos, _ypos);

    if (gpu::InputManager::isKeyDown(GLFW_KEY_C) && _timeout <= 0.00001){
        if(_type == eTrackBall){
            _type = eFirstPerson;
            _forward = glm::normalize(_center - _eye);
            _forward.y = 0.0f;
            _forward = glm::normalize(_forward);
            _angle.x += 3.141592f;
        }
        else{
            _type = eTrackBall;
            _radius = glm::length(_eye);
            glm::vec3 _eyeNorm = glm::normalize(_eye);
            _angle = glm::vec2(atan2(_eyeNorm.x, _eyeNorm.z), asin(-_eyeNorm.y) + 1.570796f);
        }
        _timeout = 0.3;
    }
    if(_timeout >= 0.0)
        _timeout -= delta_time;
    if(_type == eTrackBall){
        _eye.x = _radius * sin(_angle.y) * sin(_angle.x);
        _eye.y = _radius * cos(_angle.y);
        _eye.z = _radius * sin(_angle.y) * cos(_angle.x);

        _viewMatrix = glm::lookAt(_eye, glm::vec3(0.0f), glm::vec3(0.0, 1.0, 0.0));
    }
    else if(_type == eFirstPerson){
        glm::vec3 up = glm::vec3(0.0, 1.0, 0.0);
        glm::vec3 down = -up;
        glm::vec3 forward = glm::normalize(glm::vec3(_forward.x, 0.0f, _forward.z));
        glm::vec3 backwards = -1.f * forward;
        glm::vec3 left = glm::cross(up, forward);
        glm::vec3 right = -1.f * left;
        if (gpu::InputManager::isKeyDown(GLFW_KEY_W))
            _eye += forward * (float) delta_time * 3.f;
        if (gpu::InputManager::isKeyDown(GLFW_KEY_A)) 
            _eye += left * (float) delta_time * 3.f;
        if (gpu::InputManager::isKeyDown(GLFW_KEY_S))
            _eye += backwards * (float) delta_time * 3.f;
	    if (gpu::InputManager::isKeyDown(GLFW_KEY_D))
            _eye += right * (float) delta_time * 3.f;
        if (gpu::InputManager::isKeyDown(GLFW_KEY_LEFT_CONTROL))
            _eye += down * (float) delta_time * 3.f;
        if (gpu::InputManager::isKeyDown(GLFW_KEY_LEFT_SHIFT))
            _eye += up * (float) delta_time * 3.f;
        _radius = glm::length(_eye);

        _forward.x = sin(3.141592f - _angle.y) * sin(_angle.x);
        _forward.y = cos(3.141592f - _angle.y);
        _forward.z = sin(3.141592f - _angle.y) * cos(_angle.x);

        _viewMatrix = glm::lookAt(_eye, _eye + _forward, glm::vec3(0.0, 1.0, 0.0));
    }
}

void Camera::updateSize(uint32_t width, uint32_t height)
{
    _width = width;
    _height = height;
}

glm::mat4 Camera::getView(){
    return _viewMatrix;
}
