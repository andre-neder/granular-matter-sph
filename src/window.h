#pragma once

#include <string>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <algorithm>
#include <vector>

namespace gpu {

    struct KeyState{
        int key;
        bool enabled;
    };

    class Window{
        public:
            Window(){};
            Window(std::string title, uint32_t width, uint32_t height);
            ~Window();

            GLFWwindow* getGLFWWindow();
            bool shouldClose();
            bool wasResized(){ return m_wasResized; };
            void resizeHandled(){ m_wasResized = false; };
            void setTitle(std::string title);
            void getSize(int* width, int* height);
            void destroy();

            bool getKeyState(int key);
            bool getButtonState(int button);
            glm::vec2 getCursorPosition();
            glm::vec2 getScrollOffset();



        private:
            GLFWwindow* m_window;
            bool m_wasResized = false;
            glm::vec2 m_cursorPosition = glm::vec2(0.0);
            glm::vec2 m_scrollOffset = glm::vec2(0.0);

            std::vector<KeyState> m_keyState = {
                { GLFW_KEY_W, false },
                { GLFW_KEY_A, false },
                { GLFW_KEY_S, false },
                { GLFW_KEY_D, false },
                { GLFW_KEY_C, false },
                { GLFW_KEY_LEFT_SHIFT, false },
                { GLFW_KEY_SPACE, false },
                { GLFW_KEY_ENTER, false },
                { GLFW_KEY_RIGHT, false },
            };
             std::vector<KeyState> m_buttonState = {
                { GLFW_MOUSE_BUTTON_LEFT, false },
            };

            static void resizeCallback(GLFWwindow *window, int width, int height);
            static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
            static void mouseCallback(GLFWwindow* window, double xpos, double ypos);
            static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);
            static void mouseButtonCallback(GLFWwindow* window, int buttom, int action, int mods);


    };
}
