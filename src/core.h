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
            inline vk::Format getSwapChainImageFormat(){ return swapChainImageFormat; };
            inline size_t getSwapChainImageCount(){ return swapChainImages.size(); };
            inline vk::Extent2D getSwapChainExtent(){ return swapChainExtent; };
            inline vk::ImageView getSwapChainImageView(int index){ return swapChainImageViews[index]; };
            inline vk::SwapchainKHR getSwapChain(){ return swapChain; };
            
            QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice pDevice);
            SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice pDevice);
            
            vk::Buffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage);
            vk::Buffer bufferFromData(void* data, size_t size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage);
            void* mapBuffer(vk::Buffer buffer);
            void unmapBuffer(vk::Buffer buffer);
            void flushBuffer(vk::Buffer buffer, size_t offset, size_t size);
            void destroyBuffer(vk::Buffer buffer);
            void copyBufferToBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
            void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height);

            vk::Image createImage(vk::ImageUsageFlags imageUsage, VmaMemoryUsage memoryUsage, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling);
            void transitionImageLayout(vk::Image image, vk::Format format, vk::ImageLayout oldLayout, vk::ImageLayout newLayout);
            vk::ImageView createImageView(vk::Image image, vk::Format format);
            vk::Sampler createTextureSampler();
            void destroyImage(vk::Image image);
            void destroyImageView(vk::ImageView view);
            void destroySampler(vk::Sampler sampler);

            void createCommandPool();
            vk::CommandBuffer beginSingleTimeCommands();
            void endSingleTimeCommands(vk::CommandBuffer commandBuffer);

            vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
            vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
            vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, Window* window);
            void createSwapChain(Window* window);
            void createSwapChainImageViews();
            void destroySwapChainImageViews();
            void destroySwapChain();
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
