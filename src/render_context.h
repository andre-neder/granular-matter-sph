#pragma once
#include "core.h"

namespace gpu
{
    struct RenderSet{
        vk::Framebuffer _framebuffer;
        vk::CommandBuffer _commandBuffer;
    };

    enum RenderContextType{
        eColor,
        eColorDepth
    };

    class RenderContext
    {
    private:
        RenderContextType _type;
        Core* _core;
        vk::RenderPass _renderPass;
        // std::vector<RenderSet> _renderSets;
        std::vector<vk::Framebuffer> _framebuffers;
        std::vector<vk::CommandBuffer> _commandBuffers;
        std::vector<vk::ClearValue> _clearValues;

    public:
        RenderContext(/* args */);
        RenderContext(Core* core, RenderContextType type);
        RenderContext(Core* core, RenderContextType type, vk::AttachmentLoadOp loadOp, vk::AttachmentStoreOp storeOp);
        ~RenderContext();

        RenderContext(const RenderContext&) = delete;
        RenderContext& operator=(const RenderContext&) = delete;
        RenderContext(RenderContext&&) = default;
        RenderContext& operator=(RenderContext&&) = default;

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

