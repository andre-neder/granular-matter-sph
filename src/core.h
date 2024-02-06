#pragma once

#include "vulkan_utils.h"
#include "window.h"
#include <map>
#include <vk_mem_alloc.hpp>

#include <stb_image.h>
#include <stb_image_write.h>
#include <queue>
#include <functional>

namespace gpu
{   

    struct DescriptorSetBinding{
        uint32_t binding;
        vk::DescriptorType type;
        uint32_t count = 1;
        vk::ShaderStageFlags stages;
        vk::DescriptorBindingFlags flags = {};
        DescriptorSetBinding(uint32_t binding, vk::DescriptorType type, vk::ShaderStageFlags stages) : binding(binding), type(type), stages(stages), count(1){};
        DescriptorSetBinding(uint32_t binding, vk::DescriptorType type, uint32_t count, vk::ShaderStageFlags stages) : binding(binding), type(type), stages(stages), count(count){};
        DescriptorSetBinding(uint32_t binding, vk::DescriptorType type, uint32_t count, vk::ShaderStageFlags stages,  vk::DescriptorBindingFlags flags) : binding(binding), type(type), stages(stages), count(count), flags(flags){};
    };

    struct BufferDescriptorWrite{
        uint32_t binding;
        vk::DescriptorType type;
        vk::Buffer buffer;
        size_t size;
    };
    struct ImageDescriptorWrite{
        uint32_t binding;
        vk::DescriptorType type;
        vk::Sampler sampler;
        std::vector<vk::ImageView> imageViews;
        vk::ImageLayout imageLayout;
    };

    struct SwapChainFrame{
        vk::Image _image;
        vk::ImageView _view;
        vk::Fence _inFlight;
        vk::Semaphore _imageAvailable;
        vk::Semaphore _renderFinished;
    };

    const uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    const uint32_t MAX_QUERY_POOL_COUNT = 1024;
    class Core{
        public:
            Core(){};
            Core(bool enableValidation, Window* window);
            ~Core(){};

            void destroy();

            inline vk::Instance getInstance(){ return _instance; };
            inline vk::SurfaceKHR getSurface(){ return _surface; };
            inline vk::SurfaceFormatKHR getSurfaceFormat(){ return surfaceFormat; };
            inline vk::PhysicalDevice getPhysicalDevice(){ return physicalDevice; };
            inline vk::Device getDevice(){ return device; };
            //* Queues
            inline vk::Queue getGraphicsQueue(){ return graphicsQueue; };
            inline vk::Queue getPresentQueue(){ return presentQueue; };
            inline vk::Queue getComputeQueue(){ return computeQueue; };

            uint32_t getIdealWorkGroupSize();

            inline vma::Allocator getAllocator(){ return _allocator; };

            inline vk::CommandPool getCommandPool(){ return _commandPool; };
            //* SwapChain

            inline vk::Format getSwapChainImageFormat(){ return _swapChainImageFormat; };
            inline vk::Format getDepthFormat(){ return _depthFormat; };
            inline size_t getSwapChainImageCount(){ return _swapChainFrames.size(); };
            inline vk::Extent2D getSwapChainExtent(){ return _swapChainExtent; };
            inline vk::ImageView getSwapChainImageView(int index){ return _swapChainFrames[index]._view; };
            inline vk::ImageView getSwapChainDepthImageView(){ return _swapChainDepthImageView; };
            inline vk::SwapchainKHR getSwapChain(){ return _swapChain; };

            //* Helpers
            QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice pDevice);
            SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice pDevice);
            vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats);
            vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes);
            vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, Window* window);
            vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features);
            vk::Format findDepthFormat();
            vk::Result acquireNextImageKHR(uint32_t* imageIndex, vk::Semaphore semaphore, vk::Fence fence = VK_NULL_HANDLE);
            vk::Result presentKHR(uint32_t imageIndex, std::vector<vk::Semaphore> semaphores);
            
            //* Buffers
            vk::Buffer createBuffer(vk::DeviceSize size, vk::BufferUsageFlags bufferUsage, vma::MemoryUsage memoryUsage = vma::MemoryUsage::eAuto, vma::AllocationCreateFlags allocationFlags = {});
            vk::Buffer bufferFromData(void* data, size_t size, vk::BufferUsageFlags bufferUsage, vma::MemoryUsage memoryUsage = vma::MemoryUsage::eAuto, vma::AllocationCreateFlags allocationFlags = {});
            void updateBufferData(vk::Buffer buffer, void* data, size_t size);
            void* mapBuffer(vk::Buffer buffer);
            void unmapBuffer(vk::Buffer buffer);
            void flushBuffer(vk::Buffer buffer, size_t offset, size_t size);
            void destroyBuffer(vk::Buffer buffer);
            void copyBufferToBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size);
            void copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width = 1, uint32_t height = 1, uint32_t depth = 1);
            //* Images
            vk::Image image2DFromData(void *data, vk::ImageUsageFlags imageUsage, vma::MemoryUsage memoryUsage = vma::MemoryUsage::eAuto, vma::AllocationCreateFlags allocationFlags = {}, uint32_t width = 1, uint32_t height = 1, vk::Format format = vk::Format::eR8G8B8A8Unorm, vk::ImageTiling tiling = vk::ImageTiling::eOptimal);
            vk::Image image3DFromData(void *data, vk::ImageUsageFlags imageUsage, vma::MemoryUsage memoryUsage = vma::MemoryUsage::eAuto, vma::AllocationCreateFlags allocationFlags = {}, uint32_t width = 1, uint32_t height = 1, uint32_t depth = 1, vk::Format format = vk::Format::eR8G8B8A8Unorm, vk::ImageTiling tiling = vk::ImageTiling::eOptimal);

            vk::Image createImage2D(vk::ImageUsageFlags imageUsage, vma::MemoryUsage memoryUsage = vma::MemoryUsage::eAuto, vma::AllocationCreateFlags allocationFlags = {}, uint32_t width = 1, uint32_t height = 1, vk::Format format = vk::Format::eR8G8B8A8Unorm, vk::ImageTiling tiling = vk::ImageTiling::eOptimal);
            vk::Image createImage3D(vk::ImageUsageFlags imageUsage, vma::MemoryUsage memoryUsage = vma::MemoryUsage::eAuto, vma::AllocationCreateFlags allocationFlags = {}, uint32_t width = 1, uint32_t height = 1, uint32_t depth = 1, vk::Format format = vk::Format::eR8G8B8A8Unorm, vk::ImageTiling tiling = vk::ImageTiling::eOptimal);
            void transitionImageLayout(vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::PipelineStageFlags sourceStage, vk::PipelineStageFlags destinationStage);
            vk::ImageView createImageView2D(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags = vk::ImageAspectFlagBits::eColor);
            vk::ImageView createImageView3D(vk::Image image, vk::Format format);
            void destroyImage(vk::Image image);
            void destroyImageView(vk::ImageView view);
            //* Samplers
            vk::Sampler createSampler(vk::SamplerAddressMode addressMode = vk::SamplerAddressMode::eRepeat, vk::BorderColor borderColor = vk::BorderColor::eIntOpaqueBlack, vk::Bool32 enableAnisotropy = VK_FALSE);
            void destroySampler(vk::Sampler sampler);
            
            //* Commands
            void createCommandPool();
            vk::CommandBuffer beginSingleTimeCommands();
            void endSingleTimeCommands(vk::CommandBuffer commandBuffer);
            void beginCommands(vk::CommandBuffer commandBuffer, vk::CommandBufferBeginInfo beginInfo = vk::CommandBufferBeginInfo());
            void endCommands(vk::CommandBuffer commandBuffer);
            
            void createTimestampQueryPool(vk::QueryPool* pool);
            std::vector<uint64_t> getTimestampQueryPoolResults(vk::QueryPool* pool);

            //* Descriptors
            std::vector<vk::DescriptorSet> allocateDescriptorSets(vk::DescriptorSetLayout layout, vk::DescriptorPool pool, uint32_t count = gpu::MAX_FRAMES_IN_FLIGHT);
            void addDescriptorWrite(vk::DescriptorSet set, gpu::BufferDescriptorWrite write);
            void addDescriptorWrite(vk::DescriptorSet set, gpu::ImageDescriptorWrite write);
            void updateDescriptorSet(vk::DescriptorSet set);
            vk::DescriptorSetLayout createDescriptorSetLayout(std::vector<DescriptorSetBinding> bindings);
            vk::DescriptorPool createDescriptorPool(std::vector<vk::DescriptorPoolSize> sizes, uint32_t maxSets = 1 * gpu::MAX_FRAMES_IN_FLIGHT );
            void destroyDescriptorPool(vk::DescriptorPool pool);
            void destroyDescriptorSetLayout(vk::DescriptorSetLayout layout);
    
            vk::RenderPass createColorDepthRenderPass(vk::AttachmentLoadOp loadOp, vk::AttachmentStoreOp storeOp);

            //* Swapchain
            void createSwapChain(Window* window);
            void destroySwapChain();

            //* Shaders
            vk::ShaderModule createShaderModule(const std::vector<uint32_t> code);
            vk::ShaderModule loadShaderModule(std::string src);

            
            std::vector<SwapChainFrame> _swapChainFrames;
            
        private:
            bool m_enableValidation = true;
            std::vector<const char*> deviceExtensions = {
                VK_KHR_SWAPCHAIN_EXTENSION_NAME, 
                VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME,
                VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME,
                // VK_NV_COOPERATIVE_MATRIX_EXTENSION_NAME
            }; 

            vk::Instance _instance;
            vk::DebugUtilsMessengerEXT _debugMessenger;
            vk::SurfaceKHR _surface;
            vk::SurfaceFormatKHR surfaceFormat;
            vk::PhysicalDevice physicalDevice;
            vk::Device device;
            vk::Queue graphicsQueue;
            vk::Queue computeQueue;
            vk::Queue presentQueue;
            vma::Allocator _allocator;
            vk::CommandPool _commandPool; 


            vk::SwapchainKHR _swapChain;
            vk::Format _swapChainImageFormat;
            vk::Format _depthFormat;
            vk::Extent2D _swapChainExtent;

            vk::Image _swapChainDepthImage;
            vk::ImageView _swapChainDepthImageView;

            std::map<vk::Buffer, VmaAllocation> _bufferAllocations;
            std::map<vk::Image, VmaAllocation> _imageAllocations;
            std::map<vk::DescriptorSet, std::vector<vk::WriteDescriptorSet>> _descriptorWrites;
            std::map<vk::DescriptorSetLayout, uint32_t> _descriptorCount;
            std::map<vk::DescriptorSetLayout, vk::DescriptorBindingFlags> _descriptorBindingFlags;


            void pickPhysicalDevice();
            bool isDeviceSuitable(vk::PhysicalDevice pDevice);
            bool checkDeviceExtensionSupport(vk::PhysicalDevice pDevice);
            void createLogicalDevice();
            void createAllocator();
    };
} // namespace gpu
