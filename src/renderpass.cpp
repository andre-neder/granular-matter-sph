
#include "renderpass.h"

using namespace gpu;
   
    void RenderPass::createFramebuffers(){
        framebuffers.resize(m_core->getSwapChainImageCount());
        for (int i = 0; i < m_core->getSwapChainImageCount(); i++) {
            std::vector<vk::ImageView> attachments = {
                m_core->getSwapChainImageView(i)
            };
            vk::FramebufferCreateInfo framebufferInfo({}, renderPass, attachments, m_core->getSwapChainExtent().width, m_core->getSwapChainExtent().height, 1);
            try{
                framebuffers[i] = m_core->getDevice().createFramebuffer(framebufferInfo);
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
        }
    }

    void RenderPass::createCommandBuffers(){
        commandBuffers.resize(m_core->getSwapChainImageCount());
        vk::CommandBufferAllocateInfo allocInfo(m_core->getCommandPool(), vk::CommandBufferLevel::ePrimary, (uint32_t) commandBuffers.size());
        try{
            commandBuffers = m_core->getDevice().allocateCommandBuffers(allocInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
