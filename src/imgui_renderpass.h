#pragma once
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <functional> 
#include "core.h"

static void check_vk_result(VkResult err)
{
    if (err == 0)
        return;
    fprintf(stderr, "[vulkan] Error: VkResult = %d\n", err);
    if (err < 0)
        abort();
}

namespace gpu{
    class ImguiRenderPass{
        public:
            ImguiRenderPass(){};
            ImguiRenderPass(gpu::Core* core, gpu::Window* window);
            ~ImguiRenderPass(){};

            void initFrameResources();
            void update(int currentFrame, int imageIndex, float dt);
            void destroyFrameResources();
            void destroy(); 

            void additionalWindows();
            inline vk::CommandBuffer getCommandBuffer(int index){ return commandBuffers[index]; };

            std::function<void(int)> changeSceneCallback;
            std::function<void()> toggleWireframeCallback;

        private:
            vk::DescriptorPool descriptorPool;
            gpu::Window* m_window;
            gpu::Core* m_core;

            vk::RenderPass renderPass;
            std::vector<vk::Framebuffer> framebuffers;
            std::vector<vk::CommandBuffer> commandBuffers;
            vk::Pipeline graphicsPipeline;

            void createFramebuffers();
            void createCommandBuffers();
            
            void createDescriptorPool();
            void createRenderPass();

            vk::PhysicalDeviceProperties m_deviceProperties;
    };
}
