#pragma once

#include "spirv_utils.h"
#include "vulkan_utils.h"
#include "window.h"
#include <vk_mem_alloc.h>
#include <map>

namespace gpu
{
    class Core{
        public:
            Core(){};
            Core(bool enableValidation, Window* window);
            ~Core(){};

            void destroy();

            inline vk::Instance getInstance(){ return instance; };
            inline vk::SurfaceKHR getSurface(){ return surface; };
            inline vk::PhysicalDevice getPhysicalDevice(){ return physicalDevice; };
            inline vk::Device getDevice(){ return device; };
            inline vk::Queue getGraphicsQueue(){ return graphicsQueue; };
            inline vk::Queue getPresentQueue(){ return presentQueue; };
            inline VmaAllocator getAllocator(){ return allocator; };
            inline vk::CommandPool getCommandPool(){ return commandPool; };
            
            QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice pDevice);
            SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice pDevice);
            
            vk::Buffer Core::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage);
            void* mapBuffer(vk::Buffer buffer);
            void unmapBuffer(vk::Buffer buffer);
            void flushBuffer(vk::Buffer buffer, size_t offset, size_t size);
            void destroyBuffer(vk::Buffer buffer);
            void copyBufferToBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
            void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);
            
            void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);

            void createCommandPool();
            vk::CommandBuffer beginSingleTimeCommands();
            void endSingleTimeCommands(vk::CommandBuffer commandBuffer);
        private:
            bool m_enableValidation = true;
            std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

            vk::Instance instance;
            vk::DebugUtilsMessengerEXT m_debugMessenger;
            vk::SurfaceKHR surface;
            vk::PhysicalDevice physicalDevice;
            vk::Device device;
            vk::Queue graphicsQueue;
            vk::Queue presentQueue;
            VmaAllocator allocator;
            vk::CommandPool commandPool; 

            std::map<vk::Buffer, VmaAllocation> m_bufferAllocations;

            void pickPhysicalDevice();
            bool isDeviceSuitable(vk::PhysicalDevice pDevice);
            bool checkDeviceExtensionSupport(vk::PhysicalDevice pDevice);
            void createLogicalDevice();
            void createAllocator();
    };

    Core::Core(bool enableValidation, Window* window){
        m_enableValidation = enableValidation;
        instance = createInstance(m_enableValidation);
         if(m_enableValidation){
            m_debugMessenger = createDebugMessenger(instance);
        }
        if (glfwCreateWindowSurface(instance, window->getGLFWWindow(), nullptr, reinterpret_cast<VkSurfaceKHR*>(&surface)) != VK_SUCCESS) {
            throw std::runtime_error("failed to create window surface!");
        }
        pickPhysicalDevice();
        createLogicalDevice();
        createAllocator();

        createCommandPool();
    }

    void Core::destroy(){
        device.destroyCommandPool(commandPool);

        vmaDestroyAllocator(allocator);
        device.destroy();
        instance.destroySurfaceKHR(surface);
        if (m_enableValidation) {
            instance.destroyDebugUtilsMessengerEXT(m_debugMessenger);
        }
        instance.destroy();
    }

    void Core::pickPhysicalDevice() {
        std::vector<vk::PhysicalDevice> devices = instance.enumeratePhysicalDevices();
        bool deviceFound = false;
        for (const auto& device : devices) {
            if (isDeviceSuitable(device)) {
                deviceFound = true;
                physicalDevice = device;
                break;
            }
        }
        if(!deviceFound){
            throw std::runtime_error("failed to find a suitable GPU!");
        }
    }

    bool Core::isDeviceSuitable(vk::PhysicalDevice pDevice) {
        QueueFamilyIndices indices = findQueueFamilies(pDevice);
        bool extensionsSupported = checkDeviceExtensionSupport(pDevice);
        bool swapChainAdequate = false;
        
        if (extensionsSupported) {
            SwapChainSupportDetails swapChainSupport = querySwapChainSupport(pDevice);
            swapChainAdequate = !swapChainSupport.formats.empty() && !swapChainSupport.presentModes.empty();
        }

        vk::PhysicalDeviceFeatures supportedFeatures = pDevice.getFeatures();

        return indices.isComplete() && extensionsSupported && swapChainAdequate && supportedFeatures.samplerAnisotropy;
    }

    QueueFamilyIndices Core::findQueueFamilies(vk::PhysicalDevice pDevice) {
        QueueFamilyIndices indices;
        std::vector<vk::QueueFamilyProperties> queueFamilies = pDevice.getQueueFamilyProperties();
        for (uint32_t i = 0; i < queueFamilies.size(); i++){
            if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphicsFamily = i;
            }
            if (pDevice.getSurfaceSupportKHR(i, surface)) {
                indices.presentFamily = i;
            }
            if (indices.isComplete()) {
                break;
            }
        }
        return indices;
    }

    bool Core::checkDeviceExtensionSupport(vk::PhysicalDevice pDevice) {
        std::vector<vk::ExtensionProperties> availableExtensions = pDevice.enumerateDeviceExtensionProperties();
        std::set<std::string> requiredExtensions(deviceExtensions.begin(), deviceExtensions.end());
        for (const auto& extension : availableExtensions) {
            requiredExtensions.erase(extension.extensionName);
        }
        return requiredExtensions.empty();
    }

    SwapChainSupportDetails Core::querySwapChainSupport(vk::PhysicalDevice pDevice) {
        SwapChainSupportDetails details;
        details.capabilities = pDevice.getSurfaceCapabilitiesKHR(surface);
        details.formats = pDevice.getSurfaceFormatsKHR(surface);
        details.presentModes = pDevice.getSurfacePresentModesKHR(surface);
        return details;
    }

    void Core::createLogicalDevice(){
        QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
        std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
        std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
        float queuePriority = 1.0f;
        for (uint32_t queueFamily : uniqueQueueFamilies) {
            vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamily, 1, &queuePriority);
            queueCreateInfos.push_back(queueCreateInfo);
        }

        vk::PhysicalDeviceFeatures deviceFeatures;
        deviceFeatures.samplerAnisotropy = VK_TRUE;

        //MacOS portability extension
        std::vector<vk::ExtensionProperties> extensionProperties =  physicalDevice.enumerateDeviceExtensionProperties();
        for(auto extensionProperty : extensionProperties){
            if(std::string(extensionProperty.extensionName.data()) == std::string(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
                deviceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
        }

        vk::DeviceCreateInfo createInfo;
        if (m_enableValidation) {
            createInfo = vk::DeviceCreateInfo({}, queueCreateInfos, validationLayers, deviceExtensions, &deviceFeatures);
        }else{
            createInfo = vk::DeviceCreateInfo({}, queueCreateInfos, {}, deviceExtensions, &deviceFeatures);
        }
        try{
            device = physicalDevice.createDevice(createInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
        graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
        presentQueue = device.getQueue(indices.presentFamily.value(), 0);
    }
    void Core::createAllocator(){
        VmaAllocatorCreateInfo allocatorInfo = {};
        allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
        allocatorInfo.physicalDevice = physicalDevice;
        allocatorInfo.device = device;
        allocatorInfo.instance = instance;
        
        if(vmaCreateAllocator(&allocatorInfo, &allocator) != VK_SUCCESS)
            throw std::runtime_error("failed to create Allocator");
    }

    vk::Buffer Core::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage){
        vk::Buffer buffer;
        VmaAllocation allocation;
        vk::BufferCreateInfo bufferInfoStaging({}, size, bufferUsage);
        VmaAllocationCreateInfo allocInfoStaging = {};
        allocInfoStaging.usage = memoryUsage;
        if(vmaCreateBuffer(allocator, reinterpret_cast<VkBufferCreateInfo*>(&bufferInfoStaging), &allocInfoStaging, reinterpret_cast<VkBuffer*>(&buffer), &allocation, nullptr) != VK_SUCCESS){
            throw std::runtime_error("failed to create buffer!");
        }
        m_bufferAllocations[buffer] = allocation;
        return buffer;
    }
    void* Core::mapBuffer(vk::Buffer buffer){
        void* mappedData;
        vmaMapMemory(allocator, m_bufferAllocations[buffer], &mappedData);
        return mappedData;
    }
    void Core::unmapBuffer(vk::Buffer buffer){
        vmaUnmapMemory(allocator, m_bufferAllocations[buffer]);
    }
    void Core::destroyBuffer(vk::Buffer buffer){
        vmaDestroyBuffer(allocator, buffer, m_bufferAllocations[buffer]);
        m_bufferAllocations.erase(buffer);
    }
    void Core::flushBuffer(vk::Buffer buffer, size_t offset, size_t size){
        vmaFlushAllocation(allocator, m_bufferAllocations[buffer], offset, size);
    }
    void Core::copyBufferToBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::BufferCopy copyRegion(0, 0, size);
        commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

        endSingleTimeCommands(commandBuffer);
    }
    vk::CommandBuffer Core::beginSingleTimeCommands() {
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
    void Core::endSingleTimeCommands(vk::CommandBuffer commandBuffer) {
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

     void Core::createCommandPool() {
        QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
        vk::CommandPoolCreateInfo poolInfo({}, queueFamilyIndices.graphicsFamily.value());
        try{
            commandPool = device.createCommandPool(poolInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    void Core::copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height) {
        vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

        vk::BufferImageCopy region(0, 0, 0, vk::ImageSubresourceLayers( vk::ImageAspectFlagBits::eColor, 0, 0, 1), {0, 0, 0}, vk::Extent3D{{width, height}, 1});

        commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

        endSingleTimeCommands(commandBuffer);
    }
    void Core::transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
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
} // namespace gpu

