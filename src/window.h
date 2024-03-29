#pragma once

#include <string>
#include <GLFW/glfw3.h>

namespace gpu {

    class Window{
        public:
            Window(){};
            Window(std::string title, uint32_t width, uint32_t height);
            ~Window();

            GLFWwindow* getGLFWWindow();
            
            bool shouldClose();
            bool wasResized(){ return m_wasResized; };
            bool isMinimized();
            void resizeHandled(){ m_wasResized = false; };
            void setTitle(std::string title);
            void getSize(int* width, int* height);
            void sleep();
            void destroy();
        private:
            GLFWwindow* m_window;
            bool m_wasResized = false;
            int _width;
            int _height;

            static void resizeCallback(GLFWwindow *window, int width, int height);
            static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    };
}
