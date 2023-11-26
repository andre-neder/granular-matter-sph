#include "screenquad_renderpass.h"
#include <chrono>
#include "global.h"

using namespace gpu;

    ScreenQuadRenderPass::ScreenQuadRenderPass(gpu::Core* core){
        m_core = core;
    }
    void ScreenQuadRenderPass::init(){

        vertShaderModule = m_core->loadShaderModule(SHADER_PATH"/screenquad.vert");
        fragShaderModule = m_core->loadShaderModule(SHADER_PATH"/screenquad.frag");

        vertexBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            vertexBuffer[i] = m_core->bufferFromData((void*) vertices.data(), sizeof(Vertex) * vertices.size(), vk::BufferUsageFlagBits::eVertexBuffer, vma::MemoryUsage::eAutoPreferDevice);
        }
        indexBuffer = m_core->bufferFromData((void*) indices.data(), sizeof(uint32_t) * indices.size(), vk::BufferUsageFlagBits::eIndexBuffer, vma::MemoryUsage::eAutoPreferDevice);
        
        createDescriptorSetLayout();
        initFrameResources();
    }
    void ScreenQuadRenderPass::initFrameResources(){
        createRenderPass();
        createFramebuffers();
        createGraphicsPipeline();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
    }
    void ScreenQuadRenderPass::createRenderPass(){
         vk::AttachmentDescription colorAttachment({}, 
            m_core->getSwapChainImageFormat(), 
            vk::SampleCountFlagBits::e1, 
            vk::AttachmentLoadOp::eClear, 
            vk::AttachmentStoreOp::eStore, 
            vk::AttachmentLoadOp::eDontCare, 
            vk::AttachmentStoreOp::eDontCare, 
            vk::ImageLayout::eUndefined, 
            vk::ImageLayout::eColorAttachmentOptimal);
        vk::AttachmentReference colorAttachmentRef({}, vk::ImageLayout::eColorAttachmentOptimal);
        vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, colorAttachmentRef);
        vk::RenderPassCreateInfo renderPassInfo({}, colorAttachment, subpass);
        try{
            renderPass = m_core->getDevice().createRenderPass(renderPassInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    
    void ScreenQuadRenderPass::update(int currentFrame, int imageIndex){
        vk::CommandBufferBeginInfo beginInfo;
        try{
            commandBuffers[currentFrame].begin(beginInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
        std::vector<vk::ClearValue> clearValues = {vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
        vk::RenderPassBeginInfo renderPassInfo(renderPass, framebuffers[imageIndex], vk::Rect2D({0, 0}, m_core->getSwapChainExtent()), clearValues);
        commandBuffers[currentFrame].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
        std::vector<vk::Buffer> vertexBuffers = {vertexBuffer[currentFrame]};
        std::vector<vk::DeviceSize> offsets = {0};
        commandBuffers[currentFrame].bindVertexBuffers(0, vertexBuffers, offsets);
        commandBuffers[currentFrame].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
        commandBuffers[currentFrame].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        commandBuffers[currentFrame].endRenderPass();
        try{
            commandBuffers[currentFrame].end();
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    
    void ScreenQuadRenderPass::createDescriptorSets() {
    }
    
    void ScreenQuadRenderPass::createGraphicsPipeline(){
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo({}, vk::ShaderStageFlagBits::eVertex, vertShaderModule, "main");
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo({}, vk::ShaderStageFlagBits::eFragment, fragShaderModule, "main");
        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, bindingDescription, attributeDescriptions);
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);
        vk::Viewport viewport(0.0f, 0.0f, (float) m_core->getSwapChainExtent().width, (float) m_core->getSwapChainExtent().height, 0.0f, 1.0f);
        vk::Rect2D scissor(vk::Offset2D(0, 0),m_core->getSwapChainExtent());
        vk::PipelineViewportStateCreateInfo viewportState({}, viewport, scissor);
        vk::PipelineRasterizationStateCreateInfo rasterizer({}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f);
        vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1, VK_FALSE);
        vk::PipelineColorBlendAttachmentState colorBlendAttachment(VK_FALSE, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
        vk::PipelineColorBlendStateCreateInfo colorBlending({},VK_FALSE, vk::LogicOp::eCopy, colorBlendAttachment);
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo;

        try{
            pipelineLayout = m_core->getDevice().createPipelineLayout(pipelineLayoutInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }

        vk::GraphicsPipelineCreateInfo pipelineInfo({}, shaderStages, &vertexInputInfo, &inputAssembly, {}, &viewportState, &rasterizer, &multisampling, {}, &colorBlending, {}, pipelineLayout, renderPass);
        
        vk::Result result;
        std::tie(result, graphicsPipeline) = m_core->getDevice().createGraphicsPipeline( nullptr, pipelineInfo);
        switch ( result ){
            case vk::Result::eSuccess: break;
            default: throw std::runtime_error("failed to create graphics Pipeline!");
        }
    }

    void ScreenQuadRenderPass::createTextureImage() {
    }

    void ScreenQuadRenderPass::createUniformBuffers() {
    }

    void ScreenQuadRenderPass::createDescriptorPool() {
    }

    void ScreenQuadRenderPass::createDescriptorSetLayout() {
    }
    
    void ScreenQuadRenderPass::destroyFrameResources(){
        vk::Device device = m_core->getDevice();
        for (auto framebuffer : framebuffers) {
            device.destroyFramebuffer(framebuffer);
        }
        device.freeCommandBuffers(m_core->getCommandPool(), commandBuffers);
        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyRenderPass(renderPass);
    }

    void ScreenQuadRenderPass::destroy(){
        destroyFrameResources();

        vk::Device device = m_core->getDevice();

        device.destroyShaderModule(fragShaderModule);
        device.destroyShaderModule(vertShaderModule);

        m_core->destroyBuffer(indexBuffer);

        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            m_core->destroyBuffer(vertexBuffer[i]);
        }
    }

    void ScreenQuadRenderPass::updateUniformBuffer(uint32_t currentImage) {
    }
