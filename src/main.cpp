#pragma once

#define GLM_FORCE_RADIANS
#define GLM_ENABLE_EXPERIMENTAL
#define TINYGLTF_USE_CPP14

#ifdef _WIN32
    #include <windows.h>
#endif

// #include <optional>
// #include <set>
#include <stb_image.h>
#include <stb_image_write.h>
#include <tiny_gltf.h>
#include <chrono>

// #include <vulkan/vulkan.hpp>
// #include <GLFW/glfw3.h>
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
    vk::Instance instance;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    
    VmaAllocator allocator;

    gpu::Core core;
    gpu::Window window;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::SurfaceKHR surface;

    vk::SwapchainKHR swapChain;
    vk::Format swapChainImageFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::ImageView> swapChainImageViews;
    std::vector<vk::Framebuffer> swapChainFramebuffers;
    
    vk::RenderPass renderPass;
    vk::PipelineLayout pipelineLayout;
    vk::Pipeline graphicsPipeline;
    vk::DescriptorSetLayout descriptorSetLayout;

    vk::ShaderModule vertShaderModule;
    vk::ShaderModule fragShaderModule;

    vk::CommandPool commandPool; 
    std::vector<vk::CommandBuffer> commandBuffers;
    vk::Buffer vertexBuffer;
    VmaAllocation vertexBufferAllocation;
    vk::Buffer indexBuffer;
    VmaAllocation indexBufferAllocation;
    std::vector<vk::Buffer> uniformBuffers;
    std::vector<VmaAllocation> uniformBufferAllocations;
    vk::Image textureImage;
    VmaAllocation textureImageAllocation;
    vk::ImageView textureImageView;
    vk::Sampler textureSampler;

    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;

    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    size_t currentFrame = 0;
    // bool framebufferResized = false;

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
        instance = core.getInstance();
        surface = core.getSurface();
        physicalDevice = core.getPhysicalDevice();
        device = core.getDevice();
        graphicsQueue = core.getGraphicsQueue();
        presentQueue = core.getPresentQueue();
        allocator = core.getAllocator();
        createSwapChain();
        createImageViews();
        createRenderPass();
        createDescriptorSetLayout();
        readAndCompileShaders();
        createGraphicsPipeline();
        createFramebuffers();
        createCommandPool();
        createTextureImage();
        createTextureImageView();
        createTextureSampler();
        createVertexBuffer();
        createIndexBuffer();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
        createSyncObjects();
        initImGui();
    }

    void createAllocator(){
        VmaAllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
        allocatorInfo.physicalDevice = physicalDevice;
        allocatorInfo.device = device;
        allocatorInfo.instance = instance;
        
        if(vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS)
            throw std::runtime_error("failed to create Allocator");
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

        vk::AttachmentDescription attachment(
            {}, 
            swapChainImageFormat, 
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

        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = instance;
        init_info.PhysicalDevice = physicalDevice;
        init_info.Device = device;
        init_info.QueueFamily = core.findQueueFamilies(physicalDevice).graphicsFamily.value();
        init_info.Queue = graphicsQueue;
        init_info.PipelineCache = VK_NULL_HANDLE;
        init_info.DescriptorPool = descriptorPoolImgui;
        init_info.Allocator = VK_NULL_HANDLE;
        init_info.MinImageCount = static_cast<uint32_t>(swapChainFramebuffers.size());
        init_info.ImageCount = static_cast<uint32_t>(swapChainFramebuffers.size());
        init_info.CheckVkResultFn = check_vk_result;
        ImGui_ImplVulkan_Init(&init_info, renderPassImgui);

        vk::CommandBuffer command_buffer = beginSingleTimeCommands();
            ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
        endSingleTimeCommands(command_buffer);    

        //create command pool
        QueueFamilyIndices queueFamilyIndices = core.findQueueFamilies(physicalDevice);
        vk::CommandPoolCreateInfo commandPoolInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndices.graphicsFamily.value());
        try{
            commandPoolImgui = device.createCommandPool(commandPoolInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }

        //create commandbuffers
        commandBuffersImgui.resize(swapChainFramebuffers.size());
        vk::CommandBufferAllocateInfo allocInfo(commandPoolImgui, vk::CommandBufferLevel::ePrimary, (uint32_t) commandBuffersImgui.size());
        try{
            commandBuffersImgui = device.allocateCommandBuffers(allocInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
        //create Framebuffers
        framebuffersImgui.resize(swapChainImageViews.size());
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<vk::ImageView, 1> attachments = {
                swapChainImageViews[i]
            };
            vk::FramebufferCreateInfo framebufferInfo({}, renderPassImgui, attachments, swapChainExtent.width, swapChainExtent.height, 1);
            try{
                framebuffersImgui[i] = device.createFramebuffer(framebufferInfo);
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
        }
        //record Command Buffers

        
    }

    void recreateImGuiFramebuffer() {
        //create Framebuffers
        framebuffersImgui.resize(swapChainImageViews.size());
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::array<vk::ImageView, 1> attachments = {
                swapChainImageViews[i]
            };
            vk::FramebufferCreateInfo framebufferInfo({}, renderPassImgui, attachments, swapChainExtent.width, swapChainExtent.height, 1);
            try{
                framebuffersImgui[i] = device.createFramebuffer(framebufferInfo);
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
        }
    }

    vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
        for (const auto& availableFormat : availableFormats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
                return availableFormat;
            }
        }
        return availableFormats[0];
    }

    vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
        for (const auto& availablePresentMode : availablePresentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
                return availablePresentMode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities) {
        if (capabilities.currentExtent.width != UINT32_MAX) {
            return capabilities.currentExtent;
        } else {
            int width, height;
            window.getSize(&width, &height);
            vk::Extent2D actualExtent = {
                static_cast<uint32_t>(width),
                static_cast<uint32_t>(height)
            };
            actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
            actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
            return actualExtent;
        }
    }

    // SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice pDevice) {
    //     SwapChainSupportDetails details;
    //     details.capabilities = pDevice.getSurfaceCapabilitiesKHR(surface);
    //     details.formats = pDevice.getSurfaceFormatsKHR(surface);
    //     details.presentModes = pDevice.getSurfacePresentModesKHR(surface);
    //     return details;
    // }

    void createSwapChain() {
        SwapChainSupportDetails swapChainSupport = core.querySwapChainSupport(physicalDevice);

        vk::SurfaceFormatKHR surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
        vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
        vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities);

        uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
        if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
            imageCount = swapChainSupport.capabilities.maxImageCount;
        }

        QueueFamilyIndices indices = core.findQueueFamilies(physicalDevice);
        uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        
        vk::SwapchainCreateInfoKHR createInfo;
        createInfo.flags = {};
        createInfo.surface = surface;
        createInfo.minImageCount = imageCount;
        createInfo.imageFormat = surfaceFormat.format;
        createInfo.imageColorSpace = surfaceFormat.colorSpace;
        createInfo.imageExtent = extent;
        createInfo.imageArrayLayers = 1;
        createInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
        if (indices.graphicsFamily != indices.presentFamily) {
            createInfo.imageSharingMode = vk::SharingMode::eConcurrent;
            createInfo.queueFamilyIndexCount = 2;
            createInfo.pQueueFamilyIndices = queueFamilyIndices;
        } else {
            createInfo.imageSharingMode = vk::SharingMode::eExclusive;
        }
        createInfo.preTransform = swapChainSupport.capabilities.currentTransform;
        createInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        createInfo.presentMode = presentMode;
        createInfo.clipped = VK_TRUE;
        createInfo.oldSwapchain = VK_NULL_HANDLE;
        try{
            swapChain = device.createSwapchainKHR(createInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
        swapChainImages = device.getSwapchainImagesKHR(swapChain);
        swapChainImageFormat = surfaceFormat.format;
        swapChainExtent = extent;
    }

    void createImageViews(){
        swapChainImageViews.resize(swapChainImages.size());
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat);
        }
    }

    vk::ShaderModule createShaderModule(const std::vector<uint32_t> code) {
        vk::ShaderModuleCreateInfo createInfo({}, code);
        vk::ShaderModule shaderModule;
        try{
            shaderModule = device.createShaderModule(createInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
        return shaderModule;
    }

    void createRenderPass(){
        vk::AttachmentDescription colorAttachment({}, swapChainImageFormat, vk::SampleCountFlagBits::e1, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore, vk::AttachmentLoadOp::eDontCare, vk::AttachmentStoreOp::eDontCare, vk::ImageLayout::eUndefined, vk::ImageLayout::ePresentSrcKHR);
        vk::AttachmentReference colorAttachmentRef({}, vk::ImageLayout::eColorAttachmentOptimal);
        vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, colorAttachmentRef);
        vk::RenderPassCreateInfo renderPassInfo({}, colorAttachment, subpass);
        try{
            renderPass = device.createRenderPass(renderPassInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }

    void createDescriptorSetLayout() {
        vk::DescriptorSetLayoutBinding uboLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eVertex, nullptr);
        vk::DescriptorSetLayoutBinding samplerLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment, nullptr);

        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {uboLayoutBinding, samplerLayoutBinding};
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);

        try{
            descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }

    void readAndCompileShaders() {
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

    void createGraphicsPipeline(){
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo({}, vk::ShaderStageFlagBits::eVertex, vertShaderModule, "main");
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo({}, vk::ShaderStageFlagBits::eFragment, fragShaderModule, "main");

        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {vertShaderStageInfo, fragShaderStageInfo};

        auto bindingDescription = Vertex::getBindingDescription();
        auto attributeDescriptions = Vertex::getAttributeDescriptions();

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, bindingDescription, attributeDescriptions);
        vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);
        vk::Viewport viewport(0.0f, 0.0f, (float) swapChainExtent.width, (float) swapChainExtent.height, 0.0f, 1.0f);
        vk::Rect2D scissor(vk::Offset2D(0, 0),swapChainExtent);
        vk::PipelineViewportStateCreateInfo viewportState({}, viewport, scissor);
        vk::PipelineRasterizationStateCreateInfo rasterizer({}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f);
        vk::PipelineMultisampleStateCreateInfo multisampling({}, vk::SampleCountFlagBits::e1, VK_FALSE);
        vk::PipelineColorBlendAttachmentState colorBlendAttachment(VK_FALSE, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
        vk::PipelineColorBlendStateCreateInfo colorBlending({},VK_FALSE, vk::LogicOp::eCopy, colorBlendAttachment);
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({} ,1 , &descriptorSetLayout);

        try{
            pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }

        vk::GraphicsPipelineCreateInfo pipelineInfo({}, shaderStages, &vertexInputInfo, &inputAssembly, {}, &viewportState, &rasterizer, &multisampling, {}, &colorBlending, {}, pipelineLayout, renderPass);
        
        vk::Result result;
        std::tie(result, graphicsPipeline) = device.createGraphicsPipeline( nullptr, pipelineInfo);
        switch ( result ){
            case vk::Result::eSuccess: break;
            default: throw std::runtime_error("failed to create graphics Pipeline!");
        }
    }
    
    void createFramebuffers() {
        swapChainFramebuffers.resize(swapChainImageViews.size());
        for (size_t i = 0; i < swapChainImageViews.size(); i++) {
            std::vector<vk::ImageView> attachments = {swapChainImageViews[i]};
            vk::FramebufferCreateInfo framebufferInfo({}, renderPass, attachments, swapChainExtent.width, swapChainExtent.height, 1);
            try{
                swapChainFramebuffers[i] = device.createFramebuffer(framebufferInfo);
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
        }
    }

    void createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = core.findQueueFamilies(physicalDevice);
        vk::CommandPoolCreateInfo poolInfo({}, queueFamilyIndices.graphicsFamily.value());
        try{
            commandPool = device.createCommandPool(poolInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }

    void createTextureImage() {
        int texWidth, texHeight, texChannels;
        stbi_uc* pixels = stbi_load(RESOURCE_PATH "/checker.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
        vk::DeviceSize imageSize = texWidth * texHeight * 4;

        if (!pixels) {
            throw std::runtime_error("failed to load texture image!");
        }

        vk::Buffer stagingBuffer;
        VmaAllocation stagingBufferAllocation;
        createBuffer(stagingBuffer, stagingBufferAllocation, imageSize, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);

        void* mappedData;
        vmaMapMemory(allocator, stagingBufferAllocation, &mappedData);
            memcpy(mappedData, pixels, static_cast<size_t>(imageSize));
        vmaUnmapMemory(allocator, stagingBufferAllocation);

        stbi_image_free(pixels);

        createImage(textureImage, textureImageAllocation, vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled, VMA_MEMORY_USAGE_GPU_ONLY, texWidth, texHeight, vk::Format::eR8G8B8A8Srgb, vk::ImageTiling::eOptimal);

        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
            copyBufferToImage(stagingBuffer, textureImage, static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight));
        transitionImageLayout(textureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);

        vmaDestroyBuffer(allocator, stagingBuffer, stagingBufferAllocation);
    }

    void createTextureImageView() {
        textureImageView = createImageView(textureImage, vk::Format::eR8G8B8A8Srgb);
    }

    void createTextureSampler() {
        vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();

        vk::SamplerCreateInfo samplerInfo(
            {}, 
            vk::Filter::eLinear, 
            vk::Filter::eLinear, 
            vk::SamplerMipmapMode::eLinear, 
            vk::SamplerAddressMode::eRepeat, 
            vk::SamplerAddressMode::eRepeat, 
            vk::SamplerAddressMode::eRepeat,
            0.0f, 
            VK_TRUE, 
            properties.limits.maxSamplerAnisotropy, 
            VK_FALSE,  
            vk::CompareOp::eAlways, 
            0.0f,
            0.0f, 
            vk::BorderColor::eIntOpaqueBlack, 
            VK_FALSE 
        );
        try{
            textureSampler = device.createSampler(samplerInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }

    vk::ImageView createImageView(vk::Image image, vk::Format format) {
        vk::ImageViewCreateInfo viewInfo({}, image, vk::ImageViewType::e2D, format, {}, vk::ImageSubresourceRange( vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
        vk::ImageView imageView;
        try{
            imageView = device.createImageView(viewInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
        return imageView;
    }

    void createImage(vk::Image& image, VmaAllocation& imageAllocation, vk::ImageUsageFlags imageUsage, VmaMemoryUsage memoryUsage, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling) {
        vk::ImageCreateInfo imageInfo({}, vk::ImageType::e2D, format, vk::Extent3D{{width, height}, 1}, 1, 1, vk::SampleCountFlagBits::e1, tiling, imageUsage, vk::SharingMode::eExclusive, {}, {}, vk::ImageLayout::eUndefined);
        VmaAllocationCreateInfo allocInfoImage = {};
        allocInfoImage.usage = memoryUsage;
        if(vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&imageInfo), &allocInfoImage, reinterpret_cast<VkImage*>(&image), &imageAllocation, nullptr) != VK_SUCCESS)
            throw std::runtime_error("failed to create image!");
    }

    void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
        vk::PipelineStageFlags sourceStage;
        vk::PipelineStageFlags destinationStage;
        vk::AccessFlags srcAccessMask = {};
        vk::AccessFlags dstAccessMask = {};

        if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
            srcAccessMask = vk::AccessFlagBits::eNoneKHR;
            dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
            destinationStage = vk::PipelineStageFlagBits::eTransfer;
        } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
            srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            dstAccessMask = vk::AccessFlagBits::eShaderRead;
            sourceStage = vk::PipelineStageFlagBits::eTransfer;
            destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
        } else {
            throw std::invalid_argument("unsupported layout transition!");
        }

        vk::ImageMemoryBarrier barrier(srcAccessMask, dstAccessMask, oldLayout, newLayout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

        commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, {}, {}, 1, &barrier);

        endSingleTimeCommands(commandBuffer);
    }

    void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::BufferImageCopy region(0, 0, 0, vk::ImageSubresourceLayers( vk::ImageAspectFlagBits::eColor, 0, 0, 1), {0, 0, 0}, vk::Extent3D{{width, height}, 1});

        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }

    vk::CommandBuffer beginSingleTimeCommands() {
        vk::CommandBufferAllocateInfo allocInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1);
        vk::CommandBuffer commandBuffer;
        try{
            commandBuffer = device.allocateCommandBuffers(allocInfo)[0];
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }

        vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

        try{
            commandBuffer.begin(beginInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
        return commandBuffer;
    }

    void endSingleTimeCommands(vk::CommandBuffer commandBuffer) {
        try{
            commandBuffer.end();
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }

        vk::SubmitInfo submitInfoCopy({}, {}, commandBuffer, {});
        graphicsQueue.submit(submitInfoCopy, {});
        graphicsQueue.waitIdle();
        device.freeCommandBuffers(commandPool, 1, &commandBuffer);
    }

    void createBuffer(vk::Buffer& buffer, VmaAllocation& allocation, vk::DeviceSize size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage){
        vk::BufferCreateInfo bufferInfoStaging({}, size, bufferUsage);
        VmaAllocationCreateInfo allocInfoStaging = {};
        allocInfoStaging.usage = memoryUsage;
        if(vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfoStaging), &allocInfoStaging, reinterpret_cast<VkBuffer*>(&buffer), &allocation, nullptr) != VK_SUCCESS)
            throw std::runtime_error("failed to create buffer!");
    }

    void copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::BufferCopy copyRegion(0, 0, size);
        commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }

    void createVertexBuffer(){
        vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

        //temporary staging buffer
        vk::Buffer stagingBuffer;
        VmaAllocation stagingBufferAllocation;
        createBuffer(stagingBuffer, stagingBufferAllocation, bufferSize, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);
        
        void* mappedData;
        vmaMapMemory(allocator, stagingBufferAllocation, &mappedData);
            memcpy(mappedData, vertices.data(), (size_t) bufferSize);
        vmaUnmapMemory(allocator, stagingBufferAllocation);

        //vertexbuffer
        createBuffer(vertexBuffer, vertexBufferAllocation, bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer, VMA_MEMORY_USAGE_GPU_ONLY);
        
        //copy staging buffer data to vertexbuffer
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        //destroy staging resources
        vmaDestroyBuffer(allocator, stagingBuffer, stagingBufferAllocation);
    }

    void createIndexBuffer() {
        VkDeviceSize bufferSize = sizeof(indices[0]) * indices.size();

        //temporary staging buffer
        vk::Buffer stagingBuffer;
        VmaAllocation stagingBufferAllocation;
        createBuffer(stagingBuffer, stagingBufferAllocation, bufferSize, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);

        void* mappedData;
        vmaMapMemory(allocator, stagingBufferAllocation, &mappedData);
            memcpy(mappedData, indices.data(), (size_t) bufferSize);
        vmaUnmapMemory(allocator, stagingBufferAllocation);

        //vertexbuffer
        createBuffer(indexBuffer, indexBufferAllocation, bufferSize, vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer, VMA_MEMORY_USAGE_GPU_ONLY);

        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        vmaDestroyBuffer(allocator, stagingBuffer, stagingBufferAllocation);
    }

    void createUniformBuffers() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(swapChainImages.size());
        uniformBufferAllocations.resize(swapChainImages.size());

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            createBuffer(uniformBuffers[i], uniformBufferAllocations[i], bufferSize, vk::BufferUsageFlagBits::eUniformBuffer, VMA_MEMORY_USAGE_CPU_TO_GPU);
        }
    }

    void createDescriptorPool() {
        vk::DescriptorPoolSize poolSizeUbo(vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(swapChainImages.size()));
        vk::DescriptorPoolSize poolSizeSampler(vk::DescriptorType::eCombinedImageSampler, static_cast<uint32_t>(swapChainImages.size()));
        std::array<vk::DescriptorPoolSize, 2> poolSizes{poolSizeUbo, poolSizeSampler};

        vk::DescriptorPoolCreateInfo poolInfo({}, static_cast<uint32_t>(swapChainImages.size()), poolSizes);
        try{
            descriptorPool = device.createDescriptorPool(poolInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }

    void createDescriptorSets() {
        std::vector<vk::DescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);
        vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, static_cast<uint32_t>(swapChainImages.size()), layouts.data());
        descriptorSets.resize(swapChainImages.size());
        try{
            descriptorSets = device.allocateDescriptorSets(allocInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }

        for (size_t i = 0; i < swapChainImages.size(); i++) {

            vk::DescriptorBufferInfo bufferInfo(uniformBuffers[i], 0, sizeof(UniformBufferObject));
            vk::DescriptorImageInfo imageInfo(textureSampler, textureImageView, vk::ImageLayout::eShaderReadOnlyOptimal);
            vk::WriteDescriptorSet descriptorWriteUbo(descriptorSets[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, {}, &bufferInfo);
            vk::WriteDescriptorSet descriptorWriteSampler(descriptorSets[i], 1, 0, 1, vk::DescriptorType::eCombinedImageSampler, &imageInfo);

            std::array<vk::WriteDescriptorSet, 2> descriptorWrites{descriptorWriteUbo, descriptorWriteSampler};
            
            device.updateDescriptorSets(descriptorWrites, nullptr);
        }
    }

    void createCommandBuffers(){
        commandBuffers.resize(swapChainFramebuffers.size());
        vk::CommandBufferAllocateInfo allocInfo(commandPool, vk::CommandBufferLevel::ePrimary, (uint32_t) commandBuffers.size());
        try{
            commandBuffers = device.allocateCommandBuffers(allocInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }

        for (size_t i = 0; i < commandBuffers.size(); i++) {
            vk::CommandBufferBeginInfo beginInfo;
            try{
                commandBuffers[i].begin(beginInfo);
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
            std::vector<vk::ClearValue> clearValues = {vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f})};
            vk::RenderPassBeginInfo renderPassInfo(renderPass, swapChainFramebuffers[i], vk::Rect2D({0, 0}, swapChainExtent), clearValues);

            commandBuffers[i].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
                commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
                std::vector<vk::Buffer> vertexBuffers = {vertexBuffer};
                std::vector<vk::DeviceSize> offsets = {0};
                commandBuffers[i].bindVertexBuffers(0, vertexBuffers, offsets);
                commandBuffers[i].bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint32);
                commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);
                commandBuffers[i].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
            commandBuffers[i].endRenderPass();
            try{
                commandBuffers[i].end();
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
        }
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);
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

    void updateUniformBuffer(uint32_t currentImage) {
        static auto startTime = std::chrono::high_resolution_clock::now();

        auto currentTime = std::chrono::high_resolution_clock::now();
        float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

        UniformBufferObject ubo{};
        ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view = glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), swapChainExtent.width / (float) swapChainExtent.height, 0.1f, 10.0f);
        ubo.proj[1][1] *= -1;
        
        void* mappedData;
        vmaMapMemory(allocator, uniformBufferAllocations[currentImage], &mappedData);
            memcpy(mappedData, &ubo, (size_t) sizeof(ubo));
            vmaFlushAllocation(allocator, uniformBufferAllocations[currentImage], 0, (size_t) sizeof(ubo));
        vmaUnmapMemory(allocator, uniformBufferAllocations[currentImage]);
    }

    void drawFrame(){
        vk::Result result1 = device.waitForFences(inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex;
        vk::Result result = device.acquireNextImageKHR(swapChain, UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        if(result == vk::Result::eErrorOutOfDateKHR){
            recreateSwapChain();
            return;
        }
        else if(result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR){
            throw std::runtime_error("failed to acquire swap chain image!");
        }
        
        updateUniformBuffer(imageIndex);
        recordCommandBuffer(imageIndex);

        if ((VkFence) imagesInFlight[imageIndex] != VK_NULL_HANDLE)
            vk::Result result2 = device.waitForFences(imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        std::vector<vk::Semaphore> waitSemaphores = {imageAvailableSemaphores[currentFrame]};
        std::vector<vk::PipelineStageFlags> waitStages = {vk::PipelineStageFlagBits::eColorAttachmentOutput};
        std::vector<vk::Semaphore> signalSemaphores = {renderFinishedSemaphores[currentFrame]};
        std::array<vk::CommandBuffer, 2> submitCommandBuffers = { commandBuffers[imageIndex], commandBuffersImgui[imageIndex]};
        vk::SubmitInfo submitInfo(waitSemaphores, waitStages, submitCommandBuffers, signalSemaphores);
        device.resetFences(inFlightFences[currentFrame]);
        graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);

        std::vector<vk::SwapchainKHR> swapChains = {swapChain};
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
        for (auto framebuffer : swapChainFramebuffers) {
            device.destroyFramebuffer(framebuffer);
        }
        // imgui
        for (auto framebuffer : framebuffersImgui) {
            device.destroyFramebuffer(framebuffer);
        }
        device.freeCommandBuffers(commandPool, commandBuffers);
        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyRenderPass(renderPass);
        for (auto imageView : swapChainImageViews) {
            device.destroyImageView(imageView);
        }
        device.destroySwapchainKHR(swapChain);
        for (size_t i = 0; i < swapChainImages.size(); i++) {
            vmaDestroyBuffer(allocator, uniformBuffers[i], uniformBufferAllocations[i]);
        }
        device.destroyDescriptorPool(descriptorPool);
    }

    void cleanup(){
        cleanupSwapchain();
        // imgui
        device.freeCommandBuffers(commandPoolImgui, commandBuffersImgui);
        device.destroyRenderPass(renderPassImgui);
        device.destroyDescriptorPool(descriptorPoolImgui);

        device.destroyShaderModule(fragShaderModule);
        device.destroyShaderModule(vertShaderModule);
        device.destroySampler(textureSampler);
        device.destroyImageView(textureImageView);
        vmaDestroyImage(allocator, textureImage, textureImageAllocation);
        device.destroyDescriptorSetLayout(descriptorSetLayout);
        vmaDestroyBuffer(allocator, indexBuffer, indexBufferAllocation);
        vmaDestroyBuffer(allocator, vertexBuffer, vertexBufferAllocation);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
            device.destroySemaphore(imageAvailableSemaphores[i]);
            device.destroySemaphore(renderFinishedSemaphores[i]);
            device.destroyFence(inFlightFences[i]);
        }
        device.destroyCommandPool(commandPool);
        // imgui 
        device.destroyCommandPool(commandPoolImgui);
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();

        vmaDestroyAllocator(allocator);
        
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

        createSwapChain();
        createImageViews();
        createRenderPass();
        createGraphicsPipeline();
        createFramebuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();

        imagesInFlight.resize(swapChainImages.size(), VK_NULL_HANDLE);

        recreateImGuiFramebuffer();
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