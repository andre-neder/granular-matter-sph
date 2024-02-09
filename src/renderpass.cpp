
#include "renderpass.h"

using namespace gpu;
   
    void RenderPass::createFramebuffers(){
        framebuffers.resize(m_core->getSwapChainImageCount());
        for (int i = 0; i < m_core->getSwapChainImageCount(); i++) {
            std::vector<vk::ImageView> attachments = {
                m_core->getSwapChainImageView(i)
            };
            vk::FramebufferCreateInfo framebufferInfo({}, renderPass, attachments, m_core->getSwapChainExtent().width, m_core->getSwapChainExtent().height, 1);
            framebuffers[i] = m_core->getDevice().createFramebuffer(framebufferInfo);
        }
    }

    void RenderPass::createCommandBuffers(){
        commandBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        vk::CommandBufferAllocateInfo allocInfo(m_core->getCommandPool(), vk::CommandBufferLevel::ePrimary, (uint32_t) commandBuffers.size());
        commandBuffers = m_core->getDevice().allocateCommandBuffers(allocInfo);
    }

