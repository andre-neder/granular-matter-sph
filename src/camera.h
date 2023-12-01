#pragma once
#include <chrono>
#include <glm/glm.hpp>
#include "window.h"


namespace gpu
{   
    struct CameraData{
        glm::mat4 _viewMatrix;
        glm::mat4 _projMatrix;
    };
    class Camera
    {
    public:
        enum Type{
            eTrackBall,
            eFirstPerson
        };
        Camera();
        Camera(Type type, gpu::Window* window, uint32_t width, uint32_t height, glm::vec3 eye, glm::vec3 center);
        void update(double dt);
        void updateSize(uint32_t width, uint32_t height);
        glm::mat4 getView();
        glm::mat4 getProj();
        CameraData getCameraData();
        
    private:

        float _radius;
        glm::mat4 _viewMatrix;
        glm::mat4 _projMatrix;
        glm::vec3 _eye;
        glm::vec3 _center;
        glm::vec3 _forward;
        glm::vec2 _oldPos, _newPos;
        glm::vec2 _angle = glm::vec2(0.0f);
        uint32_t _width, _height;
        double _timeout = 0.0;
        
        gpu::Window* _window;

        Type _type;
    };
} // namespace name
