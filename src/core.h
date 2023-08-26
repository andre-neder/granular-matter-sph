#pragma once

#include "spirv_utils.h"
#include "vulkan_utils.h"
#include "window.h"
#include <vk_mem_alloc.h>

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
            
            QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice pDevice);
            SwapChainSupportDetails querySwapChainSupport(vk::PhysicalDevice pDevice);
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
    }

    void Core::destroy(){
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
} // namespace gpu

