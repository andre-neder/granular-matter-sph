#include "window.h"
#include "global.h"
#include <iostream>
extern bool simulationStepForward = false;

using namespace gpu;

void Window::resizeCallback(GLFWwindow *window, int width, int height){
    auto w = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    w->m_wasResized = true;
}
void Window::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
    auto w = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    switch (action)
    {
    case GLFW_PRESS:
        // Todo: cleanup
        if (key == GLFW_KEY_RIGHT){
            simulationStepForward = true;
        }
        if (key == GLFW_KEY_ENTER){
            simulationRunning = simulationRunning ? false : true;
        }

        // w->m_keyState.insert_or_assign (key, true);

        break;
    case GLFW_RELEASE:
        // w->m_keyState.insert_or_assign (key, false);
        break;
    default:
        break;
    }
}

void Window::mouseButtonCallback(GLFWwindow* window, int button, int action, int mods){
    auto w = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));

    switch (action)
    {
    case GLFW_PRESS:
        // w->m_buttonState.insert_or_assign (button, true);
        break;
    case GLFW_RELEASE:
        // w->m_buttonState.insert_or_assign (button, false);
        break;
    default:
        break;
    }
}

void Window::mouseCallback(GLFWwindow* window, double xpos, double ypos){
    auto w = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    w->m_cursorPosition = glm::vec2(xpos, ypos);
}

void Window::scrollCallback(GLFWwindow* window, double xoffset, double yoffset) {
    auto w = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    w->m_scrollOffset = glm::vec2(xoffset, yoffset);
}

Window::Window(std::string title, uint32_t width, uint32_t height){
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    m_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, resizeCallback);
    glfwSetKeyCallback(m_window, keyCallback);
    glfwSetMouseButtonCallback(m_window, mouseButtonCallback);
    glfwSetCursorPosCallback(m_window, mouseCallback);
    glfwSetScrollCallback(m_window, scrollCallback);

}
GLFWwindow* Window::getGLFWWindow(){
    return m_window;
}

Window::~Window(){
    
}

void Window::setTitle(std::string title){
    glfwSetWindowTitle(m_window, title.c_str());
}

void Window::getSize(int* width, int* height){
    glfwGetFramebufferSize(m_window, width, height);
}

bool Window::shouldClose(){ 
    return glfwWindowShouldClose(m_window); 
}
void Window::destroy(){
    glfwDestroyWindow(m_window);
    glfwTerminate();
}

bool Window::getKeyState(int key){
    return false;// m_keyState.at(key);
}

bool Window::getButtonState(int button){
    return false;//m_buttonState.at(button);
}

glm::vec2 Window::getCursorPosition(){
    return m_cursorPosition;
}

glm::vec2 Window::getScrollOffset(){
    return m_scrollOffset;
}