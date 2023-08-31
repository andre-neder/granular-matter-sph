#pragma once
#include "core.h"

namespace gpu
{
    class RenderPass{
        protected:
            gpu::Core* m_core;

            vk::RenderPass renderPass;
            std::vector<vk::Framebuffer> framebuffers;
            std::vector<vk::CommandBuffer> commandBuffers;
            vk::Pipeline graphicsPipeline;

            void createFramebuffers();
            void createCommandBuffers();
            virtual void createRenderPass() = 0;
        public:
            inline vk::CommandBuffer getCommandBuffer(int index){ return commandBuffers[index]; };
            virtual void initFrameResources() = 0;
            virtual void update(int imageIndex) = 0;
            virtual void destroyFrameResources() = 0;
            virtual void destroy() = 0; 
    };    
} // namespace gpu
