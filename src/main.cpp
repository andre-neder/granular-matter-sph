#pragma once

#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#define TINYGLTF_USE_CPP14

#ifdef _WIN32
    #include <windows.h>
#endif

#include <stb_image.h>
#include <stb_image_write.h>
#include <tiny_gltf.h>
#include <chrono>
#include <functional>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include "core.h"


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 2;

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static std::array<vk::VertexInputBindingDescription, 1> getBindingDescription() {
        std::array<vk::VertexInputBindingDescription, 1> bindingDescriptions = {
            vk::VertexInputBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex)
        };
        return bindingDescriptions;
    }
    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))
        };
        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{-0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{0.5f, 0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-0.5f, 0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
};

const std::vector<uint32_t> indices = {
    0, 1, 2, 2, 3, 0
};

struct UniformBufferObject {
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 proj = glm::mat4(1.0f);
};


namespace gpu{
     class BasicRenderPass{
        public:
            BasicRenderPass(){};
            BasicRenderPass(gpu::Core* core);
            ~BasicRenderPass(){};

            inline vk::CommandBuffer getCommandBuffer(int index){ return commandBuffers[index]; };
            void createRenderPass();
            void createFramebuffers();
            void createCommandBuffers();
            void readAndCompileShaders();
            void createDescriptorSets();
            void createGraphicsPipeline();
            void createTextureImage();
            void createDescriptorSetLayout();
            void updateUniformBuffer(uint32_t currentImage);
            void createUniformBuffers();
            void createDescriptorPool();
            vk::ShaderModule BasicRenderPass::createShaderModule(const std::vector<uint32_t> code); // hat hier nichts verloren
            void destroy(); 
            void update(int imageIndex);
            void init();
            void destroyFrameResources();
        private:
            gpu::Core* m_core;

            vk::RenderPass renderPass;
            std::vector<vk::Framebuffer> framebuffersDefault;
            std::vector<vk::CommandBuffer> commandBuffers;
            vk::Buffer vertexBuffer;
            vk::Buffer indexBuffer;
            std::vector<vk::Buffer> uniformBuffers;
            vk::Image textureImage;
            vk::ImageView textureImageView;
            vk::Sampler textureSampler;
            vk::DescriptorPool descriptorPool;
            std::vector<vk::DescriptorSet> descriptorSets;
            vk::PipelineLayout pipelineLayout;
            vk::Pipeline graphicsPipeline;
            vk::DescriptorSetLayout descriptorSetLayout;
            vk::ShaderModule vertShaderModule;
            vk::ShaderModule fragShaderModule;
    };
    BasicRenderPass::BasicRenderPass(gpu::Core* core){
        m_core = core;

        readAndCompileShaders();
        createTextureImage();
        textureImageView = m_core->createImageView(textureImage, vk::Format::eR8G8B8A8Srgb);
        textureSampler = m_core->createTextureSampler();
        vertexBuffer = m_core->bufferFromData((void*)vertices.data(), sizeof(vertices[0]) * vertices.size(), vk::BufferUsageFlagBits::eVertexBuffer, VMA_MEMORY_USAGE_GPU_ONLY);
        indexBuffer = m_core->bufferFromData((void*)indices.data(), sizeof(indices[0]) * indices.size(), vk::BufferUsageFlagBits::eIndexBuffer, VMA_MEMORY_USAGE_GPU_ONLY);
        createDescriptorSetLayout();
        init();
    }
    void BasicRenderPass::init(){
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
    void BasicRenderPass::createFramebuffers(){
        framebuffersDefault.resize(m_core->getSwapChainImageCount());
        for (size_t i = 0; i < m_core->getSwapChainImageCount(); i++) {
            std::vector<vk::ImageView> attachments = {
                m_core->getSwapChainImageView(i)
            };
            vk::FramebufferCreateInfo framebufferInfo({}, renderPass, attachments, m_core->getSwapChainExtent().width, m_core->getSwapChainExtent().height, 1);
            try{
                framebuffersDefault[i] = m_core->getDevice().createFramebuffer(framebufferInfo);
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
        }
    }
    void BasicRenderPass::update(int imageIndex){
        updateUniformBuffer(imageIndex);

        vk::CommandBufferBeginInfo beginInfo;
        try{
            commandBuffers[imageIndex].begin(beginInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
        std::vector<vk::ClearValue> clearValues = {vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
        vk::RenderPassBeginInfo renderPassInfo(renderPass, framebuffersDefault[imageIndex], vk::Rect2D({0, 0}, m_core->getSwapChainExtent()), clearValues);

        commandBuffers[imageIndex].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
            commandBuffers[imageIndex].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
            std::vector<vk::Buffer> vertexBuffers = {vertexBuffer};
            std::vector<vk::DeviceSize> offsets = {0};
            commandBuffers[imageIndex].bindVertexBuffers(0, vertexBuffers, offsets);
            commandBuffers[imageIndex].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
            commandBuffers[imageIndex].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[imageIndex], 0, nullptr);
            commandBuffers[imageIndex].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
        commandBuffers[imageIndex].endRenderPass();
        try{
            commandBuffers[imageIndex].end();
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    void BasicRenderPass::createCommandBuffers(){
        commandBuffers.resize(m_core->getSwapChainImageCount());
        vk::CommandBufferAllocateInfo allocInfo(m_core->getCommandPool(), vk::CommandBufferLevel::ePrimary, (uint32_t) commandBuffers.size());
        try{
            commandBuffers = m_core->getDevice().allocateCommandBuffers(allocInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    void BasicRenderPass::createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(m_core->getSwapChainImageCount(), descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, static_cast<uint32_t>(m_core->getSwapChainImageCount()), layouts.data());
        descriptorSets.resize(m_core->getSwapChainImageCount());
        try{
            descriptorSets = m_core->getDevice().allocateDescriptorSets(allocInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }

        for (size_t i = 0; i < m_core->getSwapChainImageCount(); i++) {

            vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0, sizeof(UniformBufferObject));
            vk::DescriptorImageInfo imageInfo(textureSampler, textureImageView, vk::ImageLayout::eShaderReadOnlyOptimal);
            vk::WriteDescriptorSet descriptorWriteUbo(descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, {}, &bufferInfo);
            vk::WriteDescriptorSet descriptorWriteSampler(descriptorSets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageInfo);

            std::array<vk::WriteDescriptorSet, 2> descriptorWrites{descriptorWriteUbo, descriptorWriteSampler};
            
            m_core->getDevice().updateDescriptorSets(descriptorWrites, nullptr);
        }
    }
    
    void BasicRenderPass::createGraphicsPipeline(){
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
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(RESOURCE_PATH "/checker.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }
        
        vk::Buffer stagingBuffer = m_core->createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);
        void* mappedData = m_core->mapBuffer(stagingBuffer);
        memcpy(mappedData, pixels, static_cast<size_t>(imageSize));
        m_core->unmapBuffer(stagingBuffer);

        stbi_image_free(pixels);

        textureImage = m_core->createImage(vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, VMA_MEMORY_USAGE_GPU_ONLY, texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal);
        
        m_core->transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
        m_core->copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        m_core->transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        m_core->destroyBuffer(stagingBuffer);
    }

    void BasicRenderPass::createUniformBuffers() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(m_core->getSwapChainImageCount());
        for (size_t i = 0; i < m_core->getSwapChainImageCount(); i++) {
            uniformBuffers[i] = m_core->createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
        }
    }

    void BasicRenderPass::createDescriptorPool() {
        vk::DescriptorPoolSize poolSizeUbo(vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(m_core->getSwapChainImageCount()));
        vk::DescriptorPoolSize poolSizeSampler(vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(m_core->getSwapChainImageCount()));
        std::array<vk::DescriptorPoolSize, 2> poolSizes{poolSizeUbo, poolSizeSampler};

        vk::DescriptorPoolCreateInfo poolInfo({}, static_cast<uint32_t>(m_core->getSwapChainImageCount()), poolSizes);
        try{
            descriptorPool = m_core->getDevice().createDescriptorPool(poolInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }

    void BasicRenderPass::createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
        vk::DescriptorSetLayoutBinding samplerLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr);

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);

        try{
            descriptorSetLayout = m_core->getDevice().createDescriptorSetLayout(layoutInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    void BasicRenderPass::readAndCompileShaders() {
        glslang::InitializeProcess();
        std::vector<uint32_t> vertShaderCodeSPIRV;
        std::vector<uint32_t> fragShaderCodeSPIRV;
        SpirvHelper::GLSLtoSPV(vk::ShaderStageFlagBits::eVertex, "/shader.vert", vertShaderCodeSPIRV);
        SpirvHelper::GLSLtoSPV(vk::ShaderStageFlagBits::eFragment, "/shader.frag", fragShaderCodeSPIRV);
        try{
            vertShaderModule = createShaderModule(vertShaderCodeSPIRV);
            fragShaderModule = createShaderModule(fragShaderCodeSPIRV);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
        glslang::FinalizeProcess();
    }
    vk::ShaderModule BasicRenderPass::createShaderModule(const std::vector<uint32_t> code) {
        vk::ShaderModuleCreateInfo createInfo({}, code);
        vk::ShaderModule shaderModule;
        try{
            shaderModule = m_core->getDevice().createShaderModule(createInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
        return shaderModule;
    }
    void BasicRenderPass::destroyFrameResources(){
        vk::Device device = m_core->getDevice();
        for (auto framebuffer : framebuffersDefault) {
            device.destroyFramebuffer(framebuffer);
        }
        device.freeCommandBuffers(m_core->getCommandPool(), commandBuffers);
        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyRenderPass(renderPass);
        for (size_t i = 0; i < m_core->getSwapChainImageCount(); i++) {
            m_core->destroyBuffer(uniformBuffers[i]);
        }
        device.destroyDescriptorPool(descriptorPool);
    }
    void BasicRenderPass::destroy(){
        destroyFrameResources();

        vk::Device device = m_core->getDevice();

        device.destroyShaderModule(fragShaderModule);
        device.destroyShaderModule(vertShaderModule);

        m_core->destroySampler(textureSampler);
        m_core->destroyImageView(textureImageView);
        m_core->destroyImage(textureImage);

        device.destroyDescriptorSetLayout(descriptorSetLayout);

        m_core->destroyBuffer(indexBuffer);
        m_core->destroyBuffer(vertexBuffer);
    }
    void BasicRenderPass::updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), m_core->getSwapChainExtent().width / (float) m_core->getSwapChainExtent().height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        
        void* mappedData = m_core->mapBuffer(uniformBuffers[currentImage]);
        memcpy(mappedData, &ubo, (size_t) sizeof(ubo));
        m_core->flushBuffer(uniformBuffers[currentImage], 0, (size_t) sizeof(ubo));
        m_core->unmapBuffer(uniformBuffers[currentImage]);
    }

}

class VulkanBase {
public:
        void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
private:
    bool enableValidation = true;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;

    gpu::Core core;
    gpu::Window window;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::SurfaceKHR surface;

    vk::Extent2D swapChainExtent;

    gpu::BasicRenderPass basicRenderPass;

    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    size_t currentFrame = 0;

    vk::DescriptorPool descriptorPoolImgui;
    vk::RenderPass renderPassImgui;
    vk::CommandPool commandPoolImgui;
    std::vector<vk::CommandBuffer> commandBuffersImgui;
    std::vector<vk::Framebuffer> framebuffersImgui;

    void initWindow(){
        window = gpu::Window("Application", WIDTH, HEIGHT);
    }

    void initVulkan(){
        core = gpu::Core(enableValidation, &window);
        surface = core.getSurface();
        physicalDevice = core.getPhysicalDevice();
        device = core.getDevice();
        graphicsQueue = core.getGraphicsQueue();
        presentQueue = core.getPresentQueue();
        swapChainExtent = core.getSwapChainExtent();

        basicRenderPass = gpu::BasicRenderPass(&core);
        // createRenderPass();
        // createFramebuffers();
        // createDescriptorSetLayout();
        // readAndCompileShaders();
        // createGraphicsPipeline();
        // createTextureImage();
        // createUniformBuffers();
        // createDescriptorPool();
        // createDescriptorSets();
        // createCommandBuffers();
        createSyncObjects();
        initImGui();
    }

    void createDescriptorPoolImgui(){
        std::array<vk::DescriptorPoolSize, 11> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eSampler, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eSampledImage, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageImage, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformTexelBuffer, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageTexelBuffer, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBufferDynamic, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBufferDynamic, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eInputAttachment, 1000 }
        };

        vk::DescriptorPoolCreateInfo descriptorPoolInfo({}, static_cast<uint32_t>(1000 * poolSizes.size()), poolSizes);

        try{
            descriptorPoolImgui = device.createDescriptorPool(descriptorPoolInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }

    void createRenderPassImgui(){
        vk::AttachmentDescription attachment(
            {}, 
            core.getSwapChainImageFormat(), 
            vk::SampleCountFlagBits::e1, 
            vk::AttachmentLoadOp::eLoad, 
            vk::AttachmentStoreOp::eStore, 
            vk::AttachmentLoadOp::eDontCare, 
            vk::AttachmentStoreOp::eDontCare, 
            vk::ImageLayout::eColorAttachmentOptimal, 
            vk::ImageLayout::ePresentSrcKHR
        );
        vk::AttachmentReference attachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);

        vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, attachmentRef, {}, {});

        vk::SubpassDependency dependency(
            VK_SUBPASS_EXTERNAL, 
            0, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput, 
            vk::AccessFlagBits::eNoneKHR, 
            vk::AccessFlagBits::eColorAttachmentWrite);

        std::array<vk::AttachmentDescription, 1> attachments = {attachment};

        vk::RenderPassCreateInfo renderPassInfo(
            {}, 
            attachments, 
            subpass, 
            dependency
        );
        try{
            renderPassImgui = device.createRenderPass(renderPassInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    
    void createCommandPoolImgui(){
        QueueFamilyIndices queueFamilyIndices = core.findQueueFamilies(physicalDevice);
        vk::CommandPoolCreateInfo commandPoolInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndices.graphicsFamily.value());
        try{
            commandPoolImgui = device.createCommandPool(commandPoolInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }

    void createCommandBuffersImgui(){
        commandBuffersImgui.resize(core.getSwapChainImageCount());
        vk::CommandBufferAllocateInfo allocInfo(commandPoolImgui, vk::CommandBufferLevel::ePrimary, (uint32_t) commandBuffersImgui.size());
        try{
            commandBuffersImgui = device.allocateCommandBuffers(allocInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }

    void createFramebuffersImgui(){
        framebuffersImgui.resize(core.getSwapChainImageCount());
        for (size_t i = 0; i < core.getSwapChainImageCount(); i++) {
            std::array<vk::ImageView, 1> attachments = {
                core.getSwapChainImageView(i)
            };
            vk::FramebufferCreateInfo framebufferInfo({}, renderPassImgui, attachments, core.getSwapChainExtent().width, core.getSwapChainExtent().height, 1);
            try{
                framebuffersImgui[i] = device.createFramebuffer(framebufferInfo);
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
        }
    }
    
    void initImGui() {
        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();
        ImGui_ImplGlfw_InitForVulkan(window.getGLFWWindow(), true);

        createDescriptorPoolImgui();

        createRenderPassImgui();        

        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = core.getInstance();
        init_info.PhysicalDevice = physicalDevice;
        init_info.Device = device;
        init_info.QueueFamily = core.findQueueFamilies(physicalDevice).graphicsFamily.value();
        init_info.Queue = graphicsQueue;
        init_info.PipelineCache = VK_NULL_HANDLE;
        init_info.DescriptorPool = descriptorPoolImgui;
        init_info.Allocator = VK_NULL_HANDLE;
        init_info.MinImageCount = static_cast<uint32_t>(core.getSwapChainImageCount());
        init_info.ImageCount = static_cast<uint32_t>(core.getSwapChainImageCount());
        init_info.CheckVkResultFn = check_vk_result;
        ImGui_ImplVulkan_Init(&init_info, renderPassImgui);

        vk::CommandBuffer command_buffer = core.beginSingleTimeCommands();
            ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
        core.endSingleTimeCommands(command_buffer);    

        //create command pool
        createCommandPoolImgui();
        //create commandbuffers
        createCommandBuffersImgui();
        //create Framebuffers
        createFramebuffersImgui();
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(core.getSwapChainImageCount(), VK_NULL_HANDLE);
        vk::SemaphoreCreateInfo semaphoreInfo;
        vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            try{
                imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
                renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
                inFlightFences[i] = device.createFence(fenceInfo);
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
        }
    }

    void drawFrame(){
        vk::Result result1 = device.waitForFences(inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex;
        vk::Result result;
        try{
            result = device.acquireNextImageKHR(core.getSwapChain(), UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        }
        catch(const std::exception& e){
            std::cerr << e.what() << '\n';
        }
        if(result == vk::Result::eErrorOutOfDateKHR){
            recreateSwapChain();
            return;
        }
        else if(result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR){
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        
        // updateUniformBuffer(imageIndex);
        recordCommandBuffer(imageIndex);

        if ((VkFence) imagesInFlight[imageIndex] != VK_NULL_HANDLE)
            vk::Result result2 = device.waitForFences(imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        std::vector<vk::Semaphore> waitSemaphores = {imageAvailableSemaphores[currentFrame]};
        std::vector<vk::PipelineStageFlags> waitStages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        std::vector<vk::Semaphore> signalSemaphores = {renderFinishedSemaphores[currentFrame]};
        std::array<vk::CommandBuffer, 2> submitCommandBuffers = { basicRenderPass.getCommandBuffer(imageIndex), commandBuffersImgui[imageIndex]};
        vk::SubmitInfo submitInfo(waitSemaphores, waitStages, submitCommandBuffers, signalSemaphores);
        device.resetFences(inFlightFences[currentFrame]);
        graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);

        std::vector<vk::SwapchainKHR> swapChains = {core.getSwapChain()};
        vk::PresentInfoKHR presentInfo(signalSemaphores, swapChains, imageIndex);
        try{
            result = presentQueue.presentKHR(presentInfo);
        }
        catch(const std::exception& e){
            std::cerr << e.what() << '\n';
        }
        
        if(result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || window.wasResized()){
            // framebufferResized = false;
            window.resizeHandled();
            recreateSwapChain();
        }else if(result != vk::Result::eSuccess)
            throw std::runtime_error("queue Present failed!");
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void recordCommandBuffer(uint32_t imageIndex){
        {
            vk::CommandBufferBeginInfo beginInfo;
            try{
                commandBuffersImgui[imageIndex].begin(beginInfo);
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
            std::array<vk::ClearValue, 1> clearValues{
                vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}))
            };
            vk::RenderPassBeginInfo renderPassInfo(renderPassImgui, framebuffersImgui[imageIndex], vk::Rect2D({0, 0}, swapChainExtent), clearValues);

            commandBuffersImgui[imageIndex].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

            ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffersImgui[imageIndex]);

            commandBuffersImgui[imageIndex].endRenderPass();
            try{
                commandBuffersImgui[imageIndex].end();
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
        }
        {
            basicRenderPass.update(imageIndex);
        }
    }

    void mainLoop(){
        double time = glfwGetTime();
        uint32_t fps = 0;

        bool show_demo_window = true;

        while (!window.shouldClose()) {
            glfwPollEvents();

            ImGui_ImplVulkan_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();
            ImGui::ShowDemoWindow(&show_demo_window);
            ImGui::Render();

            drawFrame();
            fps++;
            if((glfwGetTime() - time) >= 1.0){
                time = glfwGetTime();
                std::string title = "VulkanBase  FPS:"+std::to_string(fps);
                window.setTitle(title);
                fps = 0;
            }
        }
        device.waitIdle();
    }

    void cleanupSwapchain(){
       
        // imgui
        for (auto framebuffer : framebuffersImgui) {
            device.destroyFramebuffer(framebuffer);
        }
        basicRenderPass.destroyFrameResources();

        core.destroySwapChainImageViews();
        core.destroySwapChain();
        
    }

    void cleanup(){
        cleanupSwapchain();
        // imgui
        device.freeCommandBuffers(commandPoolImgui, commandBuffersImgui);
        device.destroyRenderPass(renderPassImgui);
        device.destroyDescriptorPool(descriptorPoolImgui);

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            device.destroySemaphore(imageAvailableSemaphores[i]);
            device.destroySemaphore(renderFinishedSemaphores[i]);
            device.destroyFence(inFlightFences[i]);
        }

        // imgui 
        device.destroyCommandPool(commandPoolImgui);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
        
        window.destroy();

        core.destroy();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        window.getSize(&width, &height);
        while (width == 0 || height == 0) {
            window.getSize(&width, &height);
            glfwWaitEvents();
        }
        device.waitIdle();

        cleanupSwapchain();

        core.createSwapChain(&window);
        swapChainExtent = core.getSwapChainExtent();
        core.createSwapChainImageViews();
        // createRenderPass();
        // createFramebuffers();
        // createGraphicsPipeline();
        // createUniformBuffers();
        // createDescriptorPool();
        // createDescriptorSets();
        // createCommandBuffers();
        basicRenderPass.init();

        createFramebuffersImgui();

        imagesInFlight.resize(core.getSwapChainImageCount(), VK_NULL_HANDLE);

    }
};

int main() {
    VulkanBase app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}