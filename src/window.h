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
}
