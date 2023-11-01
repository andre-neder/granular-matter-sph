#pragma once

#include "spirv_utils.h"
#include "vulkan_utils.h"
#include "window.h"
#include <vk_mem_alloc.h>
#include <map>

namespace gpu
{   
    struct DescriptorSetBinding{
        uint32_t binding;
        vk::DescriptorType type;
        vk::ShaderStageFlags stages;
    };

    struct DescriptorWrite{
        uint32_t binding;
        vk::DescriptorType type;
        vk::Buffer buffer;
        size_t size;
    };

    const uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    class Core{
        public:
            Core(){};
            Core(bool enableValidation, Window* window);
            ~Core(){};

            void destroy();

            inline vk::Instance getInstance(){ return instance; };
            inline vk::SurfaceKHR getSurface(){ return surface; };
            inline vk::SurfaceFormatKHR getSurfaceFormat(){ return surfaceFormat; };
            inline vk::PhysicalDevice getPhysicalDevice(){ return physicalDevice; };
            inline vk::Device getDevice(){ return device; };
            //* Queues
            inline vk::Queue getGraphicsQueue(){ return graphicsQueue; };
            inline vk::Queue getPresentQueue(){ return presentQueue; };
            inline vk::Queue getComputeQueue(){ return computeQueue; };

            inline VmaAllocator getAllocator(){ return allocator; };

            inline vk::CommandPool getCommandPool(){ return commandPool; };
            //* SwapChain
            inline vk::Format getSwapChainImageFormat(){ return swapChainImageFormat; };
            inline size_t getSwapChainImageCount(){ return swapChainImages.size(); };
            inline vk::Extent2D getSwapChainExtent(){ return swapChainExtent; };
            inline vk::ImageView getSwapChainImageView(int index){ return swapChainImageViews[index]; };
            inline vk::SwapchainKHR getSwapChain(){ return swapChain; };
            //* Helpers
            QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice pDevice);
            SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice pDevice);
            vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
            vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
            vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, Window* window);
            
            //* Buffers
            vk::Buffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage);
            vk::Buffer bufferFromData(void* data, size_t size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage);
            void* mapBuffer(vk::Buffer buffer);
            void unmapBuffer(vk::Buffer buffer);
            void flushBuffer(vk::Buffer buffer, size_t offset, size_t size);
            void destroyBuffer(vk::Buffer buffer);
            void copyBufferToBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
            void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);
            //* Images
            vk::Image createImage(vk::ImageUsageFlags imageUsage, VmaMemoryUsage memoryUsage, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling);
            void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
            vk::ImageView createImageView(vk::Image image, vk::Format format);
            void destroyImage(vk::Image image);
            void destroyImageView(vk::ImageView view);
            //* Samplers
            vk::Sampler createTextureSampler();
            void destroySampler(vk::Sampler sampler);
            
            //* Commands
            void createCommandPool();
            vk::CommandBuffer beginSingleTimeCommands();
            void endSingleTimeCommands(vk::CommandBuffer commandBuffer);
            
            //* Descriptors
            std::vector<vk::DescriptorSet> allocateDescriptorSets(vk::DescriptorSetLayout layout, vk::DescriptorPool pool, uint32_t count = gpu::MAX_FRAMES_IN_FLIGHT);
            void updateDescriptorSet(vk::DescriptorSet set, std::vector<gpu::DescriptorWrite> writes);
            vk::DescriptorSetLayout createDescriptorSetLayout(std::vector<DescriptorSetBinding> bindings);
            vk::DescriptorPool createDescriptorPool(std::vector<vk::DescriptorPoolSize> sizes, uint32_t maxSets = 1 * gpu::MAX_FRAMES_IN_FLIGHT );
            void destroyDescriptorPool(vk::DescriptorPool pool);
            void destroyDescriptorSetLayout(vk::DescriptorSetLayout layout);

            //* Swapchain
            void createSwapChain(Window* window);
            void createSwapChainImageViews();
            void destroySwapChainImageViews();
            void destroySwapChain();

            //* Shaders
            vk::ShaderModule createShaderModule(const std::vector<uint32_t> code);
            vk::ShaderModule loadShaderModule(std::string src);
            
        private:
            bool m_enableValidation = true;
            std::vector<const char*> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

            vk::Instance instance;
            vk::DebugUtilsMessengerEXT m_debugMessenger;
            vk::SurfaceKHR surface;
            vk::SurfaceFormatKHR surfaceFormat;
            vk::PhysicalDevice physicalDevice;
            vk::Device device;
            vk::Queue graphicsQueue;
            vk::Queue computeQueue;
            vk::Queue presentQueue;
            VmaAllocator allocator;
            vk::CommandPool commandPool; 

            vk::SwapchainKHR swapChain;
            vk::Format swapChainImageFormat;
            vk::Extent2D swapChainExtent;
            std::vector<vk::Image> swapChainImages;
            std::vector<vk::ImageView> swapChainImageViews;

            std::map<vk::Buffer, VmaAllocation> m_bufferAllocations;
            std::map<vk::Image, VmaAllocation> m_imageAllocations;

            void pickPhysicalDevice();
            bool isDeviceSuitable(vk::PhysicalDevice pDevice);
            bool checkDeviceExtensionSupport(vk::PhysicalDevice pDevice);
            void createLogicalDevice();
            void createAllocator();
    };
} // namespace gpu
