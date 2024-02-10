#pragma once
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include <functional> 
#include "core.h"
#include "render_context.h"

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

            // ImguiRenderPass(const ImguiRenderPass&) = delete;
            // ImguiRenderPass& operator=(const ImguiRenderPass&) = delete;
            // ImguiRenderPass(ImguiRenderPass&&) = default;
            // ImguiRenderPass& operator=(ImguiRenderPass&&) = default;

            void initFrameResources();
            void update(int currentFrame, int imageIndex, float dt);
            void destroyFrameResources();
            void destroy(); 

            void additionalWindows();
            inline vk::CommandBuffer getCommandBuffer(int index){ return _renderContext.getCommandBuffer(); };

            std::function<void(int)> changeSceneCallback;
            std::function<void()> toggleWireframeCallback;

        private:
            vk::DescriptorPool descriptorPool;
            gpu::Window* m_window;
            gpu::Core* _core;

            vk::Pipeline graphicsPipeline;

            gpu::RenderContext _renderContext;
            
            void createDescriptorPool();

            vk::PhysicalDeviceProperties m_deviceProperties;
    };
}
