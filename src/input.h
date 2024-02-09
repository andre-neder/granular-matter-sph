#pragma once
#include "window.h"
#include <glm/glm.hpp>
#include <array>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <functional>


namespace gpu
{   
    struct KeyBinding{
        int key;
        int action;
        std::function<void()> function;
    };

    class InputManager
    {
    private:
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
        static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

        static std::array<int, GLFW_KEY_LAST> keysDown;

        static std::vector<KeyBinding> keyBindings;
    public:
        InputManager(){};
        InputManager(Window& window);
        ~InputManager();

        void update();
        static bool isKeyDown(int key);
        static void addKeyBinding(std::function<void()>&& function, int key, int action = GLFW_PRESS);

        static glm::vec2 cursorPosition;
        static glm::vec2 scrollOffset;
    };  
} // namespace gpu



