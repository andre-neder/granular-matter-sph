#include "window.h"


using namespace gpu;

void Window::resizeCallback(GLFWwindow *window, int width, int height){
    auto w = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
    w->m_wasResized = true;
}

Window::Window(std::string title, uint32_t width, uint32_t height){
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    m_window = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, resizeCallback);
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

void gpu::Window::sleep()
{
    glfwWaitEvents();
}

bool Window::shouldClose(){ 
    return glfwWindowShouldClose(m_window); 
}
bool gpu::Window::isMinimized()
{
    getSize(&_width, &_height);
    return _width == 0 || _height == 0;
}
void Window::destroy()
{
    glfwDestroyWindow(m_window);
    glfwTerminate();
}
