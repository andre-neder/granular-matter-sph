#include "render_context.h"

using namespace gpu;


RenderContext::RenderContext(/* args */)
{
}

gpu::RenderContext::RenderContext(Core* core, RenderContextType type) : _type(type), _core(core)
{
    if(_type == RenderContextType::eColor){ // Just for imgui now
        vk::AttachmentDescription attachment(
            {}, 
            _core->getSurfaceFormat().format, 
            vk::SampleCountFlagBits::e1, 
            vk::AttachmentLoadOp::eLoad, 
            vk::AttachmentStoreOp::eStore, 
            vk::AttachmentLoadOp::eDontCare, 
            vk::AttachmentStoreOp::eDontCare, 
            vk::ImageLayout::eColorAttachmentOptimal, 
            vk::ImageLayout::ePresentSrcKHR
        );
        vk::AttachmentReference attachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);

        vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, {}, 1, &attachmentRef, {}, {}, {}, {});

        vk::SubpassDependency dependency(
            VK_SUBPASS_EXTERNAL, 
            0, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput, 
            vk::AccessFlagBits::eNoneKHR, 
            vk::AccessFlagBits::eColorAttachmentWrite);

        vk::RenderPassCreateInfo renderPassInfo(
            {}, 
            1,
            &attachment, 
            1,
            &subpass, 
            1,
            &dependency
        );

        vk::ClearValue colorClear;
        colorClear.color = vk::ClearColorValue(0.1f, 0.1f, 0.1f, 1.0f);
        _clearValues = {
            colorClear
        };

        _renderPass = _core->getDevice().createRenderPass(renderPassInfo);
    }
    else if(_type == RenderContextType::eColorDepth){

        vk::ClearValue colorClear;
        colorClear.color = vk::ClearColorValue(0.1f, 0.1f, 0.1f, 1.0f);
        vk::ClearValue depthClear;
        depthClear.depthStencil = vk::ClearDepthStencilValue(1.f);
        _clearValues = {
            colorClear, 
            depthClear
        };

        _renderPass = _core->createColorDepthRenderPass(vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore);
    }

}

gpu::RenderContext::RenderContext(Core *core, RenderContextType type, vk::AttachmentLoadOp loadOp, vk::AttachmentStoreOp storeOp)
{
    if(_type == RenderContextType::eColor){ // Just for imgui now
        vk::AttachmentDescription attachment(
            {}, 
            _core->getSurfaceFormat().format, 
            vk::SampleCountFlagBits::e1, 
            vk::AttachmentLoadOp::eLoad, 
            vk::AttachmentStoreOp::eStore, 
            vk::AttachmentLoadOp::eDontCare, 
            vk::AttachmentStoreOp::eDontCare, 
            vk::ImageLayout::eColorAttachmentOptimal, 
            vk::ImageLayout::ePresentSrcKHR
        );
        vk::AttachmentReference attachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);

        vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, {}, 1, &attachmentRef, {}, {}, {}, {});

        vk::SubpassDependency dependency(
            VK_SUBPASS_EXTERNAL, 
            0, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput, 
            vk::AccessFlagBits::eNoneKHR, 
            vk::AccessFlagBits::eColorAttachmentWrite);

        vk::RenderPassCreateInfo renderPassInfo(
            {}, 
            1,
            &attachment, 
            1,
            &subpass, 
            1,
            &dependency
        );

        vk::ClearValue colorClear;
        colorClear.color = vk::ClearColorValue(0.1f, 0.1f, 0.1f, 1.0f);
        _clearValues = {
            colorClear
        };

        _renderPass = _core->getDevice().createRenderPass(renderPassInfo);
    }
    else if(_type == RenderContextType::eColorDepth){

        vk::ClearValue colorClear;
        colorClear.color = vk::ClearColorValue(0.1f, 0.1f, 0.1f, 1.0f);
        vk::ClearValue depthClear;
        depthClear.depthStencil = vk::ClearDepthStencilValue(1.f);
        _clearValues = {
            colorClear, 
            depthClear
        };

        _renderPass = _core->createColorDepthRenderPass(loadOp, storeOp);
    }
}

RenderContext::~RenderContext()
{
}

void gpu::RenderContext::initFramebuffers()
{
    _framebuffers.resize(_core->getSwapChainImageCount());
    for (int i = 0; i < _core->getSwapChainImageCount(); i++) {
        std::vector<vk::ImageView> attachments = {
            _core->getSwapChainImageView(i)
        };
        if(_type == RenderContextType::eColorDepth){
            attachments.push_back(_core->getSwapChainDepthImageView());
        }
        vk::FramebufferCreateInfo framebufferInfo({}, _renderPass, attachments, _core->getSwapChainExtent().width, _core->getSwapChainExtent().height, 1);
        _framebuffers[i] = _core->getDevice().createFramebuffer(framebufferInfo);
    }
}

void gpu::RenderContext::initCommandBuffers()
{
    _commandBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    vk::CommandBufferAllocateInfo allocInfo(_core->getCommandPool(), vk::CommandBufferLevel::ePrimary, (uint32_t) _commandBuffers.size());
    _commandBuffers = _core->getDevice().allocateCommandBuffers(allocInfo);
}

void gpu::RenderContext::destroyFramebuffers()
{
     for (auto framebuffer : _framebuffers) {
        _core->getDevice().destroyFramebuffer(framebuffer);
    }
}

void gpu::RenderContext::freeCommandBuffers()
{
    _core->getDevice().freeCommandBuffers(_core->getCommandPool(), _commandBuffers);
}

void gpu::RenderContext::destroyRenderPass()
{
    _core->getDevice().destroyRenderPass();
}

vk::RenderPass& gpu::RenderContext::getRenderPass()
{
    return _renderPass;
}

void gpu::RenderContext::beginCommandBuffer()
{
    size_t currentFrame = _core->_swapChainContext._currentFrame;
    vk::CommandBufferBeginInfo beginInfo;
    _commandBuffers[currentFrame].begin(beginInfo);
}

vk::CommandBuffer gpu::RenderContext::getCommandBuffer()
{
    size_t currentFrame = _core->_swapChainContext._currentFrame;
    return _commandBuffers[currentFrame];
}

void gpu::RenderContext::endCommandBuffer()
{
    size_t currentFrame = _core->_swapChainContext._currentFrame;
    _commandBuffers[currentFrame].end();
}

void gpu::RenderContext::beginRenderPass(uint32_t imageIndex)
{
    size_t currentFrame = _core->_swapChainContext._currentFrame;
    vk::ClearValue colorClear;
    colorClear.color = vk::ClearColorValue(0.1f, 0.1f, 0.1f, 1.0f);
    vk::ClearValue depthClear;
    depthClear.depthStencil = vk::ClearDepthStencilValue(1.f);
    std::array<vk::ClearValue, 2> clearValues = {
        colorClear, 
        depthClear
    };
    
    vk::RenderPassBeginInfo renderPassInfo(_renderPass, _framebuffers[imageIndex], vk::Rect2D({0, 0}, _core->getSwapChainExtent()), clearValues);
    _commandBuffers[currentFrame].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    vk::Viewport viewport(0.0f, 0.0f, (float)_core->getSwapChainExtent().width, (float)_core->getSwapChainExtent().height, 0.0f, 1.0f);
    _commandBuffers[currentFrame].setViewport(0, viewport);

    vk::Rect2D scissor(vk::Offset2D(0, 0),_core->getSwapChainExtent());
    _commandBuffers[currentFrame].setScissor(0, scissor);
}

void gpu::RenderContext::endRenderPass()
{
    size_t currentFrame = _core->_swapChainContext._currentFrame;
    _commandBuffers[currentFrame].endRenderPass();
}
