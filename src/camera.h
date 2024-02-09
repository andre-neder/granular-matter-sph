#pragma once
#include <chrono>
#include <glm/glm.hpp>

#include <GLFW/glfw3.h>

namespace gpu
{

    class Camera
    {
    public:
        enum Type{
            eTrackBall,
            eFirstPerson
        };
        Camera();
        Camera(Type type, GLFWwindow* window, uint32_t width, uint32_t height, glm::vec3 eye, glm::vec3 center);
        void update(float dt);
        void updateSize(uint32_t width, uint32_t height);
        glm::mat4 getView();

    private:
        GLFWwindow* m_window;
        glm::mat4 _viewMatrix;
        glm::vec3 _eye;
        glm::vec3 _center;
        glm::vec3 _forward;
        glm::vec2 _oldPos, _newPos;
        glm::vec2 _angle = glm::vec2(0.0f);
        uint32_t _width, _height;
        double _timeout = 0.0;
        
        Type _type;

        float _radius;
    };
} // namespace gpu
