#pragma once
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
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
            void update(int imageIndex);
            void destroyFrameResources();
            void destroy(); 
        private:
            vk::DescriptorPool descriptorPool;
            
            void createRenderPass();
            void createDescriptorPool();
        
            bool show_demo_window = true;
    };
}
