#include "window.h"
#include "global.h"

extern bool simulationStepForward = false;

using namespace gpu;

void Window::resizeCallback(GLFWwindow *window, int width, int height){
    auto w = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    w->m_wasResized = true;
}
void Window::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){
    auto w = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    if (key == GLFW_KEY_RIGHT && action == GLFW_PRESS){
        simulationStepForward = true;
    }
    if (key == GLFW_KEY_SPACE && action == GLFW_PRESS){
        simulationRunning = simulationRunning ? false : true;
    }
}

Window::Window(std::string title, uint32_t width, uint32_t height){
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    m_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, resizeCallback);
    glfwSetKeyCallback(m_window, keyCallback);
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
