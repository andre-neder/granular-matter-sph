#pragma once
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <functional> 
#include "renderpass.h"

static void check_vk_result(VkResult err)
{
    if (err == 0)
        return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    if (err < 0)
        abort();
}

namespace gpu{
    class ImguiRenderPass : public RenderPass{
        public:
            ImguiRenderPass(){};
            ImguiRenderPass(gpu::Core* core, gpu::Window* window);
            ~ImguiRenderPass(){};

            void initFrameResources();
            void update(int currentFrame, int imageIndex, float dt);
            void destroyFrameResources();
            void destroy(); 

            void additionalWindows();

            std::function<void(int)> changeSceneCallback;

        private:
            vk::DescriptorPool descriptorPool;
            gpu::Window* m_window;
            
            void createDescriptorPool();
            void createRenderPass();

            vk::PhysicalDeviceProperties m_deviceProperties;
    };
}
