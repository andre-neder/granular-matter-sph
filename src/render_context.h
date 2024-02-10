#pragma once
#include "core.h"

namespace gpu
{
    struct RenderSet{
        vk::Framebuffer _framebuffer;
        vk::CommandBuffer _commandBuffer;
    };

    class RenderContext
    {
    private:
        Core* _core;
        vk::RenderPass _renderPass;
        bool _depth;
        // std::vector<RenderSet> _renderSets;
        std::vector<vk::Framebuffer> _framebuffers;
        std::vector<vk::CommandBuffer> _commandBuffers;
        std::vector<vk::ClearValue> _clearValues;

    public:
        RenderContext(/* args */);
        RenderContext(Core* core);
        RenderContext(Core* core, vk::AttachmentLoadOp loadOp, vk::AttachmentStoreOp storeOp);
        ~RenderContext();

        void initFramebuffers();
        void initCommandBuffers();
        
        void destroyFramebuffers();
        void freeCommandBuffers();

        void destroyRenderPass();
        vk::RenderPass& getRenderPass();

        void beginCommandBuffer();
        vk::CommandBuffer getCommandBuffer();
        void endCommandBuffer();

        void beginRenderPass(uint32_t imageIndex);
        void endRenderPass();
    };

} // namespace gpu

