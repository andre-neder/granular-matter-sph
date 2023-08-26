#pragma once

#include <string>
#include <GLFW/glfw3.h>
// #include <functional>


namespace gpu {

    class Window{
        public:
            Window(){};
            Window(std::string title, uint32_t width, uint32_t height);
            ~Window();

            GLFWwindow* getGLFWWindow();
            bool shouldClose();
            bool wasResized(){ return m_wasResized; };
            void resizeHandled(){ m_wasResized = false; };
            // void bindResizeCallback(std::function<void(uint16_t, uint16_t)> fn);
            void setTitle(std::string title);
            void getSize(int* width, int* height);
            void destroy();
            // vk::SurfaceKHR createSurface(vk::Instance instance);
        private:
            GLFWwindow* m_window;
            bool m_wasResized = false;

            // std::function<void(uint16_t, uint16_t)> onResize;

            static void resizeCallback(GLFWwindow *window, int width, int height);
    };

    void Window::resizeCallback(GLFWwindow *window, int width, int height){
        auto w = reinterpret_cast<Window *>(glfwGetWindowUserPointer(window));
        w->m_wasResized = true;
        // w->onResize(width, height);
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

    // void Window::bindResizeCallback(std::function<void(uint16_t, uint16_t)> fn){
    //     onResize = std::bind(fn, std::placeholders::_1, std::placeholders::_2);
    // }

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
    // vk::SurfaceKHR Window::createSurface(vk::Instance instance){
    //     vk::SurfaceKHR surface;
    //     if (glfwCreateWindowSurface(instance, m_window, nullptr, reinterpret_cast<VkSurfaceKHR*>(&surface)) != VK_SUCCESS) {
    //         throw std::runtime_error("failed to create window surface!");
    //     }
    //     return surface;
    // }
}
