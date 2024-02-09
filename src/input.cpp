#include "input.h"
#include "global.h"
#include <ranges>

using namespace gpu;

// extern bool simulationStepForward = false;

glm::vec2 InputManager::cursorPosition = glm::vec2(0.0);
glm::vec2 InputManager::scrollOffset = glm::vec2(0.0);

std::array<int, GLFW_KEY_LAST> InputManager::keysDown;
std::vector<KeyBinding> InputManager::keyBindings;

void InputManager::keyCallback(GLFWwindow *window, int key, int scancode, int action, int mods)
{
    if(key != GLFW_KEY_UNKNOWN){
        // store key down state
        keysDown[key] = (action == GLFW_PRESS || action == GLFW_REPEAT) ? true : false;

        auto bindings = keyBindings | std::views::filter([&](auto&& element){ 
            return element.key == key && element.action == action;
        } );

        for(auto&& b : bindings){
            (b.function)();
        }
    }
}

void gpu::InputManager::mouseCallback(GLFWwindow *window, double xpos, double ypos)
{
    cursorPosition = glm::vec2(xpos, ypos);
}

void gpu::InputManager::scrollCallback(GLFWwindow *window, double xoffset, double yoffset)
{
    scrollOffset = glm::vec2(xoffset, yoffset);
}

void gpu::InputManager::addKeyBinding(std::function<void()>&& function, int key, int action)
{
    keyBindings.push_back({key, action, function});
}

InputManager::InputManager(Window &window)
{
    glfwSetKeyCallback(window.getGLFWWindow(), keyCallback);
    glfwSetCursorPosCallback(window.getGLFWWindow(), mouseCallback);
    glfwSetScrollCallback(window.getGLFWWindow(), scrollCallback);
}

InputManager::~InputManager()
{
}

void gpu::InputManager::update()
{
    scrollOffset = glm::vec2(0, 0);
}

bool gpu::InputManager::isKeyDown(int key)
{
    return keysDown[key];
}
