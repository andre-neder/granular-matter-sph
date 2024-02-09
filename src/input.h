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
    enum KeyAction { 
        ePress = GLFW_PRESS, 
        eRelease = GLFW_RELEASE, 
        eHold = GLFW_REPEAT 
    };

    struct KeyBinding{
        std::string _handle;
        int _key;
        KeyAction _action;
        std::function<void()> _function;
    };

    class InputManager
    {
    private:
        static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
        static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
        static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

        static std::array<int, GLFW_KEY_LAST> keysDown;
        static std::vector<KeyBinding> keyBindings;
        static bool performKeyBindings;

    public:
        InputManager(){};
        InputManager(Window& window);
        ~InputManager();

        void update();
        static bool isKeyDown(int key);
        // add a keybinding that maps a function to a key
        static void addKeyBinding(std::string name, std::function<void()>&& function, int key, KeyAction action = KeyAction::ePress);
        // update a keybinding to use another key
        static void updateKeyBinding(std::string name, int key);
        // return the list of key bindings
        static std::vector<KeyBinding>& getKeyBindings();
        // hold any key inputs
        static void suspendKeyInput();
        static void resumeKeyInput();

        static glm::vec2 cursorPosition;
        static glm::vec2 scrollOffset;
    };  
} // namespace gpu



