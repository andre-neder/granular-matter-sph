#include "basic_renderpass.h"
#include <chrono>
#include "global.h"

using namespace gpu;

    BasicRenderPass::BasicRenderPass(gpu::Core* core){
        m_core = core;
       
    }
    void BasicRenderPass::init(){

        vertShaderModule = m_core->loadShaderModule(SHADER_PATH"/shader.vert");
        fragShaderModule = m_core->loadShaderModule(SHADER_PATH"/shader.frag");
        geomShaderModule = m_core->loadShaderModule(SHADER_PATH"/shader.geom");
        // createTextureImage();
        // textureImageView = m_core->createImageView(textureImage, vk::Format::eR8G8B8A8Srgb);
        // textureSampler = m_core->createTextureSampler();
        // vertexBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        // for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        //     vertexBuffer[i] = m_core->bufferFromData((void*)vertices.data(), sizeof(vertices[0]) * vertices.size(), vk::BufferUsageFlagBits::eVertexBuffer, VMA_MEMORY_USAGE_GPU_ONLY);
        // }
        // vertexCount = 4;
        // indexBuffer = m_core->bufferFromData((void*)indices.data(), sizeof(indices[0]) * indices.size(), vk::BufferUsageFlagBits::eIndexBuffer, VMA_MEMORY_USAGE_GPU_ONLY);
        
        createDescriptorSetLayout();
        initFrameResources();
    }
    void BasicRenderPass::initFrameResources(){
        createRenderPass();
        createFramebuffers();
        createGraphicsPipeline();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
    }
    void BasicRenderPass::createRenderPass(){
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
    
    void BasicRenderPass::update(int currentFrame, int imageIndex){
        updateUniformBuffer(currentFrame);

        void* mappedData = m_core->mapBuffer(uniformBuffersSettings[currentFrame]);
        memcpy(mappedData, &settings, (size_t) sizeof(SPHSettings));
        m_core->flushBuffer(uniformBuffersSettings[currentFrame], 0, (size_t) sizeof(SPHSettings));
        m_core->unmapBuffer(uniformBuffersSettings[currentFrame]);


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
            // commandBuffers[currentFrame].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
            // commandBuffers[currentFrame].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
            commandBuffers[currentFrame].draw(vertexCount, 1, 0, 0);

            vertexBuffers = {vertexBuffer1[currentFrame]};
            offsets = {0};
            commandBuffers[currentFrame].bindVertexBuffers(0, vertexBuffers, offsets);
            // commandBuffers[currentFrame].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
            // commandBuffers[currentFrame].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
            commandBuffers[currentFrame].draw(vertexCount1, 1, 0, 0);
        commandBuffers[currentFrame].endRenderPass();
        try{
            commandBuffers[currentFrame].end();
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    
    void BasicRenderPass::createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(gpu::MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT), layouts.data());
        descriptorSets.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        try{
            descriptorSets = m_core->getDevice().allocateDescriptorSets(allocInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }

        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {

            vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0, sizeof(UniformBufferObject));
            vk::DescriptorBufferInfo bufferInfo2(uniformBuffersSettings[i], 0, sizeof(SPHSettings));
            // vk::DescriptorImageInfo imageInfo(textureSampler, textureImageView, vk::ImageLayout::eShaderReadOnlyOptimal);
            vk::WriteDescriptorSet descriptorWriteUbo(descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, {}, &bufferInfo);
            vk::WriteDescriptorSet descriptorWriteUbo2(descriptorSets[i], 1, 0, 1, vk::DescriptorType::eUniformBuffer, {}, &bufferInfo2);
            // vk::WriteDescriptorSet descriptorWriteSampler(descriptorSets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageInfo);

            std::array<vk::WriteDescriptorSet, 2> descriptorWrites{descriptorWriteUbo, descriptorWriteUbo2};
            
            m_core->getDevice().updateDescriptorSets(descriptorWrites, nullptr);
        }
    }
    
    void BasicRenderPass::createGraphicsPipeline(){
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo({}, vk::ShaderStageFlagBits::eVertex, vertShaderModule, "main");
        vk::PipelineShaderStageCreateInfo geomShaderStageInfo({}, vk::ShaderStageFlagBits::eGeometry, geomShaderModule, "main");
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo({}, vk::ShaderStageFlagBits::eFragment, fragShaderModule, "main");

        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {vertShaderStageInfo, geomShaderStageInfo, fragShaderStageInfo};

        // auto bindingDescription = Vertex::getBindingDescription();
        // auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, bindingDescription, attributeDescriptions);
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::ePointList, VK_FALSE);
        vk::Viewport viewport(0.0f, 0.0f, (float) m_core->getSwapChainExtent().width, (float) m_core->getSwapChainExtent().height, 0.0f, 1.0f);
        vk::Rect2D scissor(vk::Offset2D(0, 0),m_core->getSwapChainExtent());
        vk::PipelineViewportStateCreateInfo viewportState({}, viewport, scissor);
        vk::PipelineRasterizationStateCreateInfo rasterizer({}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f);
        vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1, VK_FALSE);
        vk::PipelineColorBlendAttachmentState colorBlendAttachment(VK_FALSE, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
        vk::PipelineColorBlendStateCreateInfo colorBlending({},VK_FALSE, vk::LogicOp::eCopy, colorBlendAttachment);
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({} ,1 , &descriptorSetLayout);

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

    void BasicRenderPass::createTextureImage() {
        // int texWidth, texHeight, texChannels;
        // stbi_uc* pixels = stbi_load(RESOURCE_PATH "/checker.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        // vk::DeviceSize imageSize = texWidth * texHeight * 4;

        // if (!pixels) {
        //     throw std::runtime_error("failed to load texture image!");
        // }
        
        // vk::Buffer stagingBuffer = m_core->createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);
        // void* mappedData = m_core->mapBuffer(stagingBuffer);
        // memcpy(mappedData, pixels, static_cast<size_t>(imageSize));
        // m_core->unmapBuffer(stagingBuffer);

        // stbi_image_free(pixels);

        // textureImage = m_core->createImage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, VMA_MEMORY_USAGE_GPU_ONLY, texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal);
        
        // m_core->transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        // m_core->copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        // m_core->transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        // m_core->destroyBuffer(stagingBuffer);
    }

    void BasicRenderPass::createUniformBuffers() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            uniformBuffers[i] = m_core->createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
        }

        vk::DeviceSize bufferSize2 = sizeof(SPHSettings);

        uniformBuffersSettings.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            uniformBuffersSettings[i] = m_core->createBuffer(bufferSize2, vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
        }
    }

    void BasicRenderPass::createDescriptorPool() {
        vk::DescriptorPoolSize poolSizeUbo(vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        vk::DescriptorPoolSize poolSizeSettings(vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        // vk::DescriptorPoolSize poolSizeSampler(vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        std::array<vk::DescriptorPoolSize, 2> poolSizes{poolSizeUbo, poolSizeSettings};

        vk::DescriptorPoolCreateInfo poolInfo({}, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT), poolSizes);
        try{
            descriptorPool = m_core->getDevice().createDescriptorPool(poolInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }

    void BasicRenderPass::createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
        vk::DescriptorSetLayoutBinding settingsLayoutBinding(1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, nullptr);
        // vk::DescriptorSetLayoutBinding samplerLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr);

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, settingsLayoutBinding};
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);

        try{
            descriptorSetLayout = m_core->getDevice().createDescriptorSetLayout(layoutInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    
    void BasicRenderPass::destroyFrameResources(){
        vk::Device device = m_core->getDevice();
        for (auto framebuffer : framebuffers) {
            device.destroyFramebuffer(framebuffer);
        }
        device.freeCommandBuffers(m_core->getCommandPool(), commandBuffers);
        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            m_core->destroyBuffer(uniformBuffers[i]);
            m_core->destroyBuffer(uniformBuffersSettings[i]);
        }
        device.destroyDescriptorPool(descriptorPool);
        device.destroyRenderPass(renderPass);
    }
    void BasicRenderPass::destroy(){
        destroyFrameResources();

        vk::Device device = m_core->getDevice();

        device.destroyShaderModule(fragShaderModule);
        device.destroyShaderModule(vertShaderModule);
        device.destroyShaderModule(geomShaderModule);

        // m_core->destroySampler(textureSampler);
        // m_core->destroyImageView(textureImageView);
        // m_core->destroyImage(textureImage);
        
        device.destroyDescriptorSetLayout(descriptorSetLayout);

        // m_core->destroyBuffer(indexBuffer);

        // for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        //     m_core->destroyBuffer(vertexBuffer[i]);
        // }
    }
    void BasicRenderPass::updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();
        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::mat4(1.0f),// glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(0.0f, 0.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), m_core->getSwapChainExtent().width / (float) m_core->getSwapChainExtent().height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        
        void* mappedData = m_core->mapBuffer(uniformBuffers[currentImage]);
        memcpy(mappedData, &ubo, (size_t) sizeof(ubo));
        m_core->flushBuffer(uniformBuffers[currentImage], 0, (size_t) sizeof(ubo));
        m_core->unmapBuffer(uniformBuffers[currentImage]);
    }
