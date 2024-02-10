#include "core.h"
#include "shader_utils.h"
#include <iostream>
#include <sstream>
#include <fstream>

#define MAX_VARIABLE_DESCRIPTOR_COUNT 32

using namespace gpu;

bool hasStencilComponent(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint;
}

bool QueueFamilyIndices::isComplete() {
	return graphicsFamily.has_value() && presentFamily.has_value() && computeFamily.has_value();
}

std::string trim(const std::string& line)
{
    const char* WhiteSpace = " \t\v\r\n";
    std::size_t start = line.find_first_not_of(WhiteSpace);
    std::size_t end = line.find_last_not_of(WhiteSpace);
    return start == end ? std::string() : line.substr(start, end - start + 1);
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    std::cerr << "[validation] " << trim(pCallbackData->pMessage).c_str() << std::endl;
    return VK_FALSE;
}


Core::Core(bool enableValidation, Window* window){
    _enableValidation = enableValidation;
    createInstance();

    if(_enableValidation){
        createDebugMessenger();
    }

    // if (glfwCreateWindowSurface(*_instance, window->getGLFWWindow(), nullptr, reinterpret_cast<VkSurfaceKHR*>(&_surface)) != VK_SUCCESS) {
    //     throw std::runtime_error("failed to create window surface!");
    // }
    
    VkSurfaceKHR surfaceTmp;
    VkResult err = glfwCreateWindowSurface(*_instance, window->getGLFWWindow(), nullptr, &surfaceTmp);

    _surface = vk::UniqueSurfaceKHR(surfaceTmp, *_instance);

    pickPhysicalDevice();
    createLogicalDevice();
    createAllocator();
    createCommandPool();
    createSwapChain(window);
}

uint32_t gpu::Core::getIdealWorkGroupSize()
{
    uint32_t vendorID = _physicalDevice.getProperties().vendorID;
    uint32_t workGroupSize = 1;
    switch (vendorID)
    {
    case 4318: // nvidia
        workGroupSize = 32;
        break;
    case 4130: // AMD
        workGroupSize = 64;
        break;
    case 32902: // Intel
        workGroupSize = 16;
        break; 
    default:
        break;
    }
    return workGroupSize;
}

void Core::pickPhysicalDevice() {
    std::vector<vk::PhysicalDevice> devices = _instance->enumeratePhysicalDevices();
    bool deviceFound = false;
    for (const auto& device : devices) {
        if (isDeviceSuitable(device)) {
            deviceFound = true;
            _physicalDevice = device;
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

    auto m_deviceFeatures2 = pDevice.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceBufferDeviceAddressFeatures, vk::PhysicalDeviceDescriptorIndexingFeatures, vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>();
    bool supportsAllFeatures =
        m_deviceFeatures2.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
        m_deviceFeatures2.get<vk::PhysicalDeviceFeatures2>().features.geometryShader &&
        m_deviceFeatures2.get<vk::PhysicalDeviceFeatures2>().features.fillModeNonSolid &&
        m_deviceFeatures2.get<vk::PhysicalDeviceFeatures2>().features.shaderSampledImageArrayDynamicIndexing &&
        m_deviceFeatures2.get<vk::PhysicalDeviceRayTracingPipelineFeaturesKHR>().rayTracingPipeline &&
        m_deviceFeatures2.get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>().accelerationStructure &&
        m_deviceFeatures2.get<vk::PhysicalDeviceBufferDeviceAddressFeatures>().bufferDeviceAddress &&
        m_deviceFeatures2.get<vk::PhysicalDeviceDescriptorIndexingFeatures>().runtimeDescriptorArray;
        m_deviceFeatures2.get<vk::PhysicalDeviceDescriptorIndexingFeatures>().shaderSampledImageArrayNonUniformIndexing && 
        m_deviceFeatures2.get<vk::PhysicalDeviceDescriptorIndexingFeatures>().descriptorBindingVariableDescriptorCount && 
        m_deviceFeatures2.get<vk::PhysicalDeviceDescriptorIndexingFeatures>().descriptorBindingPartiallyBound &&
        m_deviceFeatures2.get<vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>().shaderBufferFloat32Atomics &&
        m_deviceFeatures2.get<vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>().shaderBufferFloat32AtomicAdd;

    return indices.isComplete() && extensionsSupported && swapChainAdequate && supportsAllFeatures;
}

QueueFamilyIndices Core::findQueueFamilies(vk::PhysicalDevice pDevice) {
    QueueFamilyIndices indices;
    std::vector<vk::QueueFamilyProperties> queueFamilies = pDevice.getQueueFamilyProperties();
    for (uint32_t i = 0; i < queueFamilies.size(); i++){
        if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eGraphics && queueFamilies[i].timestampValidBits > 0) {
            indices.graphicsFamily = i;
        }
        if (queueFamilies[i].queueFlags & vk::QueueFlagBits::eCompute && queueFamilies[i].timestampValidBits > 0) {
            indices.computeFamily = i;
        }
        if (pDevice.getSurfaceSupportKHR(i, *_surface)) {
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
    std::set<std::string> requiredExtensions(_deviceExtensions.begin(), _deviceExtensions.end());
    for (const auto& extension : availableExtensions) {
        requiredExtensions.erase(extension.extensionName);
    }
    return requiredExtensions.empty();
}

bool gpu::Core::checkValidationLayerSupport()
{
    std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();
	for (const char* layerName : _validationLayers) {
		bool layerFound = false;
		for (const auto& layerProperties : availableLayers) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}
		if (!layerFound) {
			return false;
		}
	}
	return true;
}

SwapChainSupportDetails Core::querySwapChainSupport(vk::PhysicalDevice pDevice) {
    SwapChainSupportDetails details;
    details.capabilities = pDevice.getSurfaceCapabilitiesKHR(*_surface);
    details.formats = pDevice.getSurfaceFormatsKHR(*_surface);
    details.presentModes = pDevice.getSurfacePresentModesKHR(*_surface);
    return details;
}

void Core::createLogicalDevice(){
    QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    std::set<uint32_t> uniqueQueueFamilies = {indices.graphicsFamily.value(), indices.presentFamily.value()};
    float queuePriority = 1.0f;
    for (uint32_t queueFamily : uniqueQueueFamilies) {
        vk::DeviceQueueCreateInfo queueCreateInfo({}, queueFamily, 1, &queuePriority);
        queueCreateInfos.push_back(queueCreateInfo);
    }

    //MacOS portability extension
    std::vector<vk::ExtensionProperties> extensionProperties =  _physicalDevice.enumerateDeviceExtensionProperties();
    for(auto extensionProperty : extensionProperties){
        if(std::string(extensionProperty.extensionName.data()) == std::string(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
            _deviceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }

    vk::DeviceCreateInfo deviceCreateInfo;
	if (_enableValidation)
	{
		deviceCreateInfo = vk::DeviceCreateInfo({}, queueCreateInfos, _validationLayers, _deviceExtensions, {});
	}
	else
	{
		deviceCreateInfo = vk::DeviceCreateInfo({}, queueCreateInfos, {}, _deviceExtensions, {});
	}
    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceBufferDeviceAddressFeatures, vk::PhysicalDeviceDescriptorIndexingFeatures, vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT> deviceFeatureCreateInfo = {
		deviceCreateInfo,
		vk::PhysicalDeviceFeatures2().setFeatures(vk::PhysicalDeviceFeatures().setSamplerAnisotropy(true).setGeometryShader(true).setShaderSampledImageArrayDynamicIndexing(true).setFillModeNonSolid(true)),
		vk::PhysicalDeviceRayTracingPipelineFeaturesKHR().setRayTracingPipeline(true),
		vk::PhysicalDeviceAccelerationStructureFeaturesKHR().setAccelerationStructure(true),
		vk::PhysicalDeviceBufferDeviceAddressFeatures().setBufferDeviceAddress(true),
		vk::PhysicalDeviceDescriptorIndexingFeatures().setRuntimeDescriptorArray(true).setShaderSampledImageArrayNonUniformIndexing(true).setDescriptorBindingVariableDescriptorCount(true).setDescriptorBindingPartiallyBound(true),
        vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT().setShaderBufferFloat32Atomics(true).setShaderBufferFloat32AtomicAdd(true)
	};

    _device = _physicalDevice.createDeviceUnique(deviceFeatureCreateInfo.get<vk::DeviceCreateInfo>());


    graphicsQueue = _device->getQueue(indices.graphicsFamily.value(), 0);
    computeQueue = _device->getQueue(indices.computeFamily.value(), 0);
    presentQueue = _device->getQueue(indices.presentFamily.value(), 0);
}

void Core::createAllocator(){
    vma::AllocatorCreateInfo allocatorInfo;
    allocatorInfo.flags = vma::AllocatorCreateFlagBits::eBufferDeviceAddress;
    allocatorInfo.physicalDevice = _physicalDevice;
    allocatorInfo.device = *_device;
    allocatorInfo.instance = *_instance;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;

    _allocator = vma::createAllocatorUnique(allocatorInfo);
}

void gpu::Core::createInstance()
{
    if (_enableValidation && !checkValidationLayerSupport()) {
		throw std::runtime_error("validation layers requested, but not available!");
	}
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

    extensions.push_back("VK_EXT_debug_utils");
    extensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);

	vk::ApplicationInfo applicationInfo("VulkanBase", VK_MAKE_VERSION(0, 0 ,1), "VulkanEngine", 1, VK_API_VERSION_1_1);

    _instance = vk::createInstanceUnique(vk::InstanceCreateInfo{
        vk::InstanceCreateFlags{ VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR }, &applicationInfo, 
        static_cast<uint32_t>(_validationLayers.size()), _validationLayers.data(),
        static_cast<uint32_t>(extensions.size()), extensions.data() 
    });

}
vk::DispatchLoaderDynamic Core::_dispatchLoaderDynamic;
void gpu::Core::createDebugMessenger()
{
  
    Core::_dispatchLoaderDynamic = vk::DispatchLoaderDynamic(*_instance, vkGetInstanceProcAddr);

    _debugMessenger = _instance->createDebugUtilsMessengerEXTUnique(
        vk::DebugUtilsMessengerCreateInfoEXT{ {},
            vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo,
            vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
            debugCallback },
        nullptr, _dispatchLoaderDynamic);
}

vk::Buffer Core::createBuffer(vk::DeviceSize size, vk::BufferUsageFlags bufferUsage, vma::MemoryUsage memoryUsage, vma::AllocationCreateFlags allocationFlags){
    vk::Buffer buffer;
    vma::Allocation allocation;

    vk::BufferCreateInfo bufferInfo;
	bufferInfo.size = size;
	bufferInfo.usage = bufferUsage;

    vma::AllocationCreateInfo bufferAllocInfo;
	bufferAllocInfo.usage = memoryUsage;
	bufferAllocInfo.flags = allocationFlags;

    std::tie(buffer, allocation) = _allocator->createBuffer(bufferInfo, bufferAllocInfo);

    _bufferAllocations[buffer] = allocation;
    return buffer;
}

vk::Buffer Core::bufferFromData(void* data, size_t size, vk::BufferUsageFlags bufferUsage, vma::MemoryUsage memoryUsage, vma::AllocationCreateFlags allocationFlags){
    if(memoryUsage != vma::MemoryUsage::eAutoPreferHost && memoryUsage != vma::MemoryUsage::eAutoPreferDevice && memoryUsage != vma::MemoryUsage::eAuto){
        throw std::exception("Unsupported memory usage.");
    }

    bool hostAccess = ((memoryUsage == vma::MemoryUsage::eAutoPreferDevice || memoryUsage == vma::MemoryUsage::eAuto) && allocationFlags == vma::AllocationCreateFlagBits::eHostAccessSequentialWrite) || memoryUsage == vma::MemoryUsage::eAutoPreferHost;

    vk::Buffer stagingBuffer;
    if(memoryUsage == vma::MemoryUsage::eAutoPreferDevice){
        stagingBuffer = createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc, vma::MemoryUsage::eAuto, vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
    }
    else if(hostAccess){
        stagingBuffer = createBuffer(size, bufferUsage, memoryUsage, allocationFlags);
    }
    
    void* mappedData = mapBuffer(stagingBuffer);
    memcpy(mappedData, data, (size_t) size);
    unmapBuffer(stagingBuffer);

    if(hostAccess){
        return stagingBuffer;
    }

    vk::Buffer buffer = createBuffer(size, vk::BufferUsageFlagBits::eTransferDst | bufferUsage, memoryUsage, allocationFlags);
    copyBufferToBuffer(stagingBuffer, buffer, size);
    destroyBuffer(stagingBuffer);

    return buffer;
}
void gpu::Core::updateBufferData(vk::Buffer buffer, void *data, size_t size)
{

    vk::Buffer stagingBuffer = createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc, vma::MemoryUsage::eAuto, vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);

    void* mappedData = mapBuffer(stagingBuffer);
    memcpy(mappedData, data, (size_t) size);
    unmapBuffer(stagingBuffer);

    copyBufferToBuffer(stagingBuffer, buffer, size);
    destroyBuffer(stagingBuffer);
}
void *Core::mapBuffer(vk::Buffer buffer)
{
    void* mappedData = _allocator->mapMemory(_bufferAllocations[buffer]);
    return mappedData;
}
void Core::unmapBuffer(vk::Buffer buffer){
    _allocator->unmapMemory(_bufferAllocations[buffer]);
}
void Core::destroyBuffer(vk::Buffer buffer){
    _allocator->destroyBuffer(buffer, _bufferAllocations[buffer]);
    _bufferAllocations.erase(buffer);
}
void Core::flushBuffer(vk::Buffer buffer, size_t offset, size_t size){
    _allocator->flushAllocation(_bufferAllocations[buffer], offset, size);
}
void Core::copyBufferToBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
    vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

    vk::BufferCopy copyRegion(0, 0, size);
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
}
vk::CommandBuffer Core::beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo(*_commandPool, vk::CommandBufferLevel::ePrimary, 1);
    vk::CommandBuffer commandBuffer = _device->allocateCommandBuffers(allocInfo)[0];

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);

    commandBuffer.begin(beginInfo);

    return commandBuffer;
}
void Core::endSingleTimeCommands(vk::CommandBuffer commandBuffer) {

    commandBuffer.end();

    vk::SubmitInfo submitInfoCopy({}, {}, commandBuffer, {});
    graphicsQueue.submit(submitInfoCopy, {});
    graphicsQueue.waitIdle();
    _device->freeCommandBuffers(*_commandPool, 1, &commandBuffer);
}

void gpu::Core::beginCommands(vk::CommandBuffer commandBuffer, vk::CommandBufferBeginInfo beginInfo)
{
    commandBuffer.begin(beginInfo);
}

void gpu::Core::endCommands(vk::CommandBuffer commandBuffer)
{
    commandBuffer.end();
}

void gpu::Core::createTimestampQueryPool(vk::QueryPool* pool)
{
    vk::QueryPoolCreateInfo createInfo{
        {},
        vk::QueryType::eTimestamp,
        MAX_QUERY_POOL_COUNT, 
        {}
    };
    
    vk::Result result = _device->createQueryPool(&createInfo, nullptr, pool);
    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error("Failed to create time query pool!");
    }

}


std::vector<uint64_t> gpu::Core::getTimestampQueryPoolResults(vk::QueryPool *pool)
{
    uint64_t buffer[MAX_QUERY_POOL_COUNT];

    vk::Result result = _device->getQueryPoolResults(*pool, 0, MAX_QUERY_POOL_COUNT, sizeof(uint64_t) * MAX_QUERY_POOL_COUNT, buffer, sizeof(uint64_t), vk::QueryResultFlagBits::e64);
    if (result == vk::Result::eNotReady)
    {
        return std::vector<uint64_t>(buffer, buffer + MAX_QUERY_POOL_COUNT);
    }
    else if (result == vk::Result::eSuccess)
    {
        return std::vector<uint64_t>(buffer, buffer + MAX_QUERY_POOL_COUNT);
    }
    else
    {
        throw std::runtime_error("Failed to receive query results!");
    }

}

std::vector<vk::DescriptorSet> gpu::Core::allocateDescriptorSets(vk::DescriptorSetLayout layout, vk::DescriptorPool pool, uint32_t count)
{
    //Duplicate layout for each set
    std::vector<vk::DescriptorSetLayout> layouts(count, layout);

    vk::DescriptorSetAllocateInfo allocInfo(pool, count, layouts.data());
    auto descriptorCount = _descriptorCount.at(layout);
    std::vector<uint32_t> counts(count, descriptorCount);
    vk::DescriptorSetVariableDescriptorCountAllocateInfo variableDescriptorCountAllocInfo{count, counts.data()};
    allocInfo.pNext = &variableDescriptorCountAllocInfo;

   
    auto sets = _device->allocateDescriptorSets(allocInfo);
    for(auto& set : sets){
        //* create a write queue for each set
        _descriptorWrites[set] = std::vector<vk::WriteDescriptorSet>();
    }
    return sets;

}

void gpu::Core::updateDescriptorSet(vk::DescriptorSet set)
{
    auto& descriptorWrites = _descriptorWrites.at(set);
    _device->updateDescriptorSets(descriptorWrites, nullptr);
    for(auto descriptorWrite : descriptorWrites){
        delete descriptorWrite.pBufferInfo;
        delete descriptorWrite.pImageInfo;
    }
    descriptorWrites.clear();
}


void Core::addDescriptorWrite(vk::DescriptorSet set, gpu::BufferDescriptorWrite write)
{
    vk::DescriptorBufferInfo* bufferInfo = new vk::DescriptorBufferInfo(write.buffer, 0, write.size);
    vk::WriteDescriptorSet descriptorWrite(set, write.binding, 0, 1, write.type, {}, bufferInfo);
    _descriptorWrites.at(set).push_back(descriptorWrite);
}

void Core::addDescriptorWrite(vk::DescriptorSet set, gpu::ImageDescriptorWrite write)
{
    std::vector<vk::DescriptorImageInfo>* imageInfos = new std::vector<vk::DescriptorImageInfo>(); 
    if(write.type == vk::DescriptorType::eSampler){
        imageInfos->push_back(vk::DescriptorImageInfo(write.sampler, {}, {}));
    }
    else if(write.type == vk::DescriptorType::eSampledImage){
        for(auto view : write.imageViews){
            imageInfos->push_back(vk::DescriptorImageInfo({}, view, write.imageLayout));
        }
    }
    else if(write.type == vk::DescriptorType::eCombinedImageSampler){
        for(auto view : write.imageViews){
            imageInfos->push_back(vk::DescriptorImageInfo(write.sampler, view, write.imageLayout));
        }
    }
    else{
        throw "Unsupported write type";
    }
    vk::WriteDescriptorSet descriptorWrite(set, write.binding, 0, (uint32_t)imageInfos->size(), write.type, imageInfos->data());
    _descriptorWrites.at(set).push_back(descriptorWrite);
}



vk::DescriptorSetLayout gpu::Core::createDescriptorSetLayout(std::vector<DescriptorSetBinding> bindings)
{
    std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;
    std::vector<vk::DescriptorBindingFlags> flags;
    vk::DescriptorBindingFlags layoutFlags;
    // std::vector<uint32_t> descriptorCounts;
    uint32_t descriptorCount = 0;
    for(auto b : bindings){
        descriptorSetLayoutBindings.push_back(vk::DescriptorSetLayoutBinding(b.binding, b.type, b.count, b.stages, nullptr));
        flags.push_back(b.flags);
        layoutFlags |= b.flags;
        descriptorCount = std::max(descriptorCount, b.count);
    }

    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, (uint32_t)descriptorSetLayoutBindings.size(), descriptorSetLayoutBindings.data());

    vk::DescriptorSetLayoutBindingFlagsCreateInfo bindingFlags{(uint32_t)flags.size(), flags.data()};
    layoutInfo.pNext = &bindingFlags;

    auto descriptorSetLayout = _device->createDescriptorSetLayout(layoutInfo);
    _descriptorCount[descriptorSetLayout] = descriptorCount;
    _descriptorBindingFlags[descriptorSetLayout] = layoutFlags;
    return descriptorSetLayout;

}

vk::DescriptorPool gpu::Core::createDescriptorPool(std::vector<vk::DescriptorPoolSize> sizes, uint32_t maxSets)
{
    vk::DescriptorPoolCreateInfo poolInfo({}, maxSets, sizes);
    return _device->createDescriptorPool(poolInfo);
}

void gpu::Core::destroyDescriptorPool(vk::DescriptorPool pool)
{
    _device->destroyDescriptorPool(pool);
}

void gpu::Core::destroyDescriptorSetLayout(vk::DescriptorSetLayout layout)
{
    _device->destroyDescriptorSetLayout(layout);
}

vk::RenderPass gpu::Core::createColorDepthRenderPass(vk::AttachmentLoadOp loadOp, vk::AttachmentStoreOp storeOp)
{
    vk::AttachmentDescription colorAttachment(
        {}, 
        getSwapChainImageFormat(), 
        vk::SampleCountFlagBits::e1, 
        loadOp, 
        storeOp, 
        vk::AttachmentLoadOp::eDontCare, 
        vk::AttachmentStoreOp::eDontCare, 
        loadOp == vk::AttachmentLoadOp::eLoad ? vk::ImageLayout::eColorAttachmentOptimal : vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal
    );
    vk::AttachmentReference colorAttachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);

    vk::AttachmentDescription depthAttachment(
        {}, 
        getDepthFormat(), 
        vk::SampleCountFlagBits::e1, 
        loadOp, 
        storeOp, 
        loadOp, 
        vk::AttachmentStoreOp::eDontCare, 
        loadOp == vk::AttachmentLoadOp::eLoad ? vk::ImageLayout::eDepthStencilAttachmentOptimal : vk::ImageLayout::eUndefined , 
        vk::ImageLayout::eDepthStencilAttachmentOptimal
    );
    vk::AttachmentReference depthAttachmentRef(1, vk::ImageLayout::eDepthStencilAttachmentOptimal);

    std::array<vk::AttachmentDescription, 2> attachments{
        colorAttachment,
        depthAttachment
    };

    vk::SubpassDescription subpass;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    vk::RenderPassCreateInfo renderPassInfo({}, attachments, subpass);

    return _device->createRenderPass(renderPassInfo);
}

void Core::createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(_physicalDevice);
    vk::CommandPoolCreateInfo poolInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndices.graphicsFamily.value());
    _commandPool = _device->createCommandPoolUnique(poolInfo);
}

void Core::copyBufferToImage(vk::Buffer buffer, vk::Image image, uint32_t width, uint32_t height, uint32_t depth) {
    vk::CommandBuffer commandBuffer = beginSingleTimeCommands();

    vk::BufferImageCopy region(0, 0, 0, vk::ImageSubresourceLayers( vk::ImageAspectFlagBits::eColor, 0, 0, 1), {0, 0, 0}, vk::Extent3D{{width, height}, depth});

    commandBuffer.copyBufferToImage(buffer, image, vk::ImageLayout::eTransferDstOptimal, 1, &region);

    endSingleTimeCommands(commandBuffer);
}

vk::Image Core::image2DFromData(void *data, vk::ImageUsageFlags imageUsage, vma::MemoryUsage memoryUsage, vma::AllocationCreateFlags allocationFlags, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling)
{
    uint32_t formatSize = 0;
    switch (format){
        case vk::Format::eR32G32Sfloat:
            formatSize = 2 * 4;
            break;
        case vk::Format::eR32G32B32Sfloat:
            formatSize = 3 * 4;
            break;
        case vk::Format::eR32G32B32A32Sfloat:
            formatSize = 4 * 4;
            break;
        case vk::Format::eR8G8B8A8Unorm:
            formatSize = 1 * 4;
            break;
        default:
            break;
    }

    vk::Buffer stagingBuffer = bufferFromData(data, width * height * formatSize, vk::BufferUsageFlagBits::eTransferSrc, vma::MemoryUsage::eAuto, vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
        
    vk::Image image = createImage2D(vk::ImageUsageFlagBits::eTransferDst | imageUsage, memoryUsage, allocationFlags, width, height, format, tiling);
        
    transitionImageLayout(image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer);
    copyBufferToImage(stagingBuffer, image, static_cast<uint32_t>(width), static_cast<uint32_t>(height));
    transitionImageLayout(image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader);

    destroyBuffer(stagingBuffer);
    
    return image;
}

void Core::transitionImageLayout(vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout, vk::PipelineStageFlags sourceStage, vk::PipelineStageFlags destinationStage)
{
    vk::CommandBuffer commandBuffer = beginSingleTimeCommands();
    vk::AccessFlags srcAccessMask = {};
    vk::AccessFlags dstAccessMask = {};

    switch (oldLayout)
    {
        case vk::ImageLayout::eUndefined:
            srcAccessMask = vk::AccessFlagBits::eNone;
            break;
        case vk::ImageLayout::ePreinitialized:
            srcAccessMask = vk::AccessFlagBits::eHostWrite;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            srcAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            srcAccessMask = vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            srcAccessMask = vk::AccessFlagBits::eTransferRead;
            break;
        case vk::ImageLayout::eTransferDstOptimal:
            srcAccessMask = vk::AccessFlagBits::eTransferWrite;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            srcAccessMask = vk::AccessFlagBits::eShaderRead;
            break;
        default:
            throw std::invalid_argument("unsupported layout transition!");
    }

    switch (newLayout)
    {
        case vk::ImageLayout::eTransferDstOptimal:
            dstAccessMask = vk::AccessFlagBits::eTransferWrite;
            break;
        case vk::ImageLayout::eTransferSrcOptimal:
            dstAccessMask = vk::AccessFlagBits::eTransferRead;
            break;
        case vk::ImageLayout::eColorAttachmentOptimal:
            dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite;
            break;
        case vk::ImageLayout::eDepthStencilAttachmentOptimal:
            dstAccessMask = dstAccessMask | vk::AccessFlagBits::eDepthStencilAttachmentWrite;
            break;
        case vk::ImageLayout::eShaderReadOnlyOptimal:
            if (srcAccessMask == vk::AccessFlagBits::eNone)
            {
                srcAccessMask = vk::AccessFlagBits::eHostWrite | vk::AccessFlagBits::eTransferWrite;
            }
            dstAccessMask = vk::AccessFlagBits::eShaderRead;
            break;
        default:
            throw std::invalid_argument("unsupported layout transition!");
    }


    vk::ImageMemoryBarrier barrier(srcAccessMask, dstAccessMask, oldLayout, newLayout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, {}, {}, 1, &barrier);

    endSingleTimeCommands(commandBuffer);
}


vk::Image gpu::Core::image3DFromData(void *data, vk::ImageUsageFlags imageUsage, vma::MemoryUsage memoryUsage, vma::AllocationCreateFlags allocationFlags, uint32_t width, uint32_t height, uint32_t depth, vk::Format format, vk::ImageTiling tiling)
{
    uint32_t formatSize = 0;
    switch (format){
        case vk::Format::eR32G32Sfloat:
            formatSize = 2 * 4;
            break;
        case vk::Format::eR32G32B32Sfloat:
            formatSize = 3 * 4;
            break;
        case vk::Format::eR32G32B32A32Sfloat:
            formatSize = 4 * 4;
            break;
        default:
            break;
    }

    vk::Buffer stagingBuffer = bufferFromData(data, width * height * depth * formatSize, vk::BufferUsageFlagBits::eTransferSrc, vma::MemoryUsage::eAuto, vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
        
    vk::Image image = createImage3D(vk::ImageUsageFlagBits::eTransferDst | imageUsage, memoryUsage, allocationFlags, width, height, depth, format, tiling);
        
    transitionImageLayout(image, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, vk::PipelineStageFlagBits::eTopOfPipe, vk::PipelineStageFlagBits::eTransfer);
    copyBufferToImage(stagingBuffer, image, static_cast<uint32_t>(width), static_cast<uint32_t>(height), static_cast<uint32_t>(depth));
    transitionImageLayout(image, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eFragmentShader);

    destroyBuffer(stagingBuffer);
    
    return image;
}

vk::Image Core::createImage2D(vk::ImageUsageFlags imageUsage, vma::MemoryUsage memoryUsage, vma::AllocationCreateFlags allocationFlags, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling) {
    vk::Image image;
    vma::Allocation allocation;

    vk::ImageCreateInfo imageInfo;
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = format;
    imageInfo.extent = vk::Extent3D{{width, height}, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.tiling = tiling;
    imageInfo.usage = imageUsage;

    vma::AllocationCreateInfo imageAllocInfo;
    imageAllocInfo.usage = memoryUsage;
	imageAllocInfo.flags = allocationFlags;

    std::tie(image, allocation) = _allocator->createImage(imageInfo, imageAllocInfo);

    _imageAllocations[image] = allocation;
    return image;
}

vk::Image Core::createImage3D(vk::ImageUsageFlags imageUsage, vma::MemoryUsage memoryUsage, vma::AllocationCreateFlags allocationFlags, uint32_t width, uint32_t height, uint32_t depth, vk::Format format, vk::ImageTiling tiling) {
    vk::Image image;
    vma::Allocation allocation;
    
    vk::ImageCreateInfo imageInfo;
    imageInfo.imageType = vk::ImageType::e3D;
    imageInfo.format = format;
    imageInfo.extent = vk::Extent3D{{width, height}, depth};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.tiling = tiling;
    imageInfo.usage = imageUsage;

    vma::AllocationCreateInfo imageAllocInfo;
	imageAllocInfo.usage = memoryUsage;
	imageAllocInfo.flags = allocationFlags;

    std::tie(image, allocation) = _allocator->createImage(imageInfo, imageAllocInfo);

    _imageAllocations[image] = allocation;
    return image;
}

void Core::destroyImage(vk::Image image){
    _allocator->destroyImage(image, _imageAllocations[image]);
    _imageAllocations.erase(image);
}

void Core::destroyImageView(vk::ImageView view){
    _device->destroyImageView(view);
}

vk::ImageView Core::createImageView2D(vk::Image image, vk::Format format, vk::ImageAspectFlags aspectFlags) {
    vk::ImageViewCreateInfo viewInfo({}, image, vk::ImageViewType::e2D, format, {}, vk::ImageSubresourceRange( aspectFlags, 0, 1, 0, 1));
    return _device->createImageView(viewInfo);
}

vk::ImageView Core::createImageView3D(vk::Image image, vk::Format format) {
    vk::ImageViewCreateInfo viewInfo({}, image, vk::ImageViewType::e3D, format, {}, vk::ImageSubresourceRange( vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    return _device->createImageView(viewInfo);;
}

vk::Sampler Core::createSampler(vk::SamplerAddressMode addressMode, vk::BorderColor borderColor, vk::Bool32 enableAnisotropy) {

    vk::PhysicalDeviceProperties properties = _physicalDevice.getProperties();

    vk::SamplerCreateInfo samplerInfo(
        {}, 
        vk::Filter::eLinear, 
        vk::Filter::eLinear, 
        vk::SamplerMipmapMode::eLinear, 
        addressMode, 
        addressMode, 
        addressMode,
        0.0f, 
        enableAnisotropy, 
        properties.limits.maxSamplerAnisotropy, 
        VK_FALSE,  
        vk::CompareOp::eAlways, 
        0.0f,
        0.0f, 
        borderColor, 
        VK_FALSE 
    );

    return _device->createSampler(samplerInfo);
}
void Core::destroySampler(vk::Sampler sampler){
    _device->destroySampler(sampler);
}

vk::Framebuffer gpu::Core::createFramebuffer(vk::RenderPass renderPass, vk::ArrayProxy<vk::ImageView> attachments)
{
    vk::FramebufferCreateInfo framebufferInfo({}, renderPass, attachments, getSwapChainExtent().width, getSwapChainExtent().height, 1);
    return _device->createFramebuffer(framebufferInfo);

}

std::vector<vk::Framebuffer> gpu::Core::createColorFramebuffer(vk::RenderPass renderPass)
{
    std::vector<vk::Framebuffer> framebuffers;
    framebuffers.resize(getSwapChainImageCount());
    for (int i = 0; i < getSwapChainImageCount(); i++) {
        std::vector<vk::ImageView> attachments = {
            getSwapChainImageView(i)
        };
        vk::FramebufferCreateInfo framebufferInfo({}, renderPass, attachments, getSwapChainExtent().width, getSwapChainExtent().height, 1);
        framebuffers[i] = _device->createFramebuffer(framebufferInfo);
    }
    return framebuffers;
}

std::vector<vk::Framebuffer> gpu::Core::createColorDepthFramebuffer(vk::RenderPass renderPass)
{
    std::vector<vk::Framebuffer> framebuffers;
    framebuffers.resize(getSwapChainImageCount());
    for (int i = 0; i < getSwapChainImageCount(); i++) {
        std::vector<vk::ImageView> attachments = {
            getSwapChainImageView(i),
            getSwapChainDepthImageView()
        };
        vk::FramebufferCreateInfo framebufferInfo({}, renderPass, attachments, getSwapChainExtent().width, getSwapChainExtent().height, 1);
        framebuffers[i] = _device->createFramebuffer(framebufferInfo);
    }
    return framebuffers;
}

vk::SurfaceFormatKHR Core::chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
    for (const auto& availableFormat : availableFormats) {
        if (availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return availableFormat;
        }
    }
    return availableFormats[0];
}

vk::PresentModeKHR Core::chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
    for (const auto& availablePresentMode : availablePresentModes) {
        if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
            return availablePresentMode;
        }
    }
    return vk::PresentModeKHR::eFifo;
}

vk::Extent2D Core::chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, Window* window) {
    if (capabilities.currentExtent.width != UINT32_MAX) {
        return capabilities.currentExtent;
    } else {
        int width, height;
        window->getSize(&width, &height);
        vk::Extent2D actualExtent = {
            static_cast<uint32_t>(width),
            static_cast<uint32_t>(height)
        };
        actualExtent.width = std::clamp(actualExtent.width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        actualExtent.height = std::clamp(actualExtent.height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
        return actualExtent;
    }
}

void Core::createSwapChain(Window* window) {
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(_physicalDevice);

    _surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities, window);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    QueueFamilyIndices indices = findQueueFamilies(_physicalDevice);
    uint32_t queueFamilyIndices[] = {indices.graphicsFamily.value(), indices.presentFamily.value()};
    
    vk::SwapchainCreateInfoKHR createInfo;
    createInfo.flags = {};
    createInfo.surface = *_surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = _surfaceFormat.format;
    createInfo.imageColorSpace = _surfaceFormat.colorSpace;
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

    _swapChainContext = {};
    _swapChainContext._swapChain = _device->createSwapchainKHR(createInfo);
    _swapChainContext._imageFormat = _surfaceFormat.format;
    _swapChainContext._extent = extent;

    std::vector<vk::Image> swapChainImages = _device->getSwapchainImagesKHR(_swapChainContext._swapChain);

    _swapChainContext._frames.resize(swapChainImages.size());
    for (size_t i = 0; i < getSwapChainImageCount(); i++) {
        _swapChainContext._frames[i] = {};
        _swapChainContext._frames[i]._image = swapChainImages[i];
        _swapChainContext._frames[i]._view = createImageView2D(swapChainImages[i], _swapChainContext._imageFormat);

        vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);
        _swapChainContext._frames[i]._inFlight = _device->createFence(fenceInfo);

        vk::SemaphoreCreateInfo semaphoreInfo;
        _swapChainContext._frames[i]._imageAvailable = _device->createSemaphore(semaphoreInfo);
        _swapChainContext._frames[i]._renderFinished = _device->createSemaphore(semaphoreInfo);
    }

    _swapChainContext._depthFormat = findDepthFormat();
    _swapChainDepthImage = createImage2D(vk::ImageUsageFlagBits::eDepthStencilAttachment, vma::MemoryUsage::eAutoPreferDevice, {},_swapChainContext._extent.width, _swapChainContext._extent.height, _swapChainContext._depthFormat);
    _swapChainDepthImageView = createImageView2D(_swapChainDepthImage, _swapChainContext._depthFormat, vk::ImageAspectFlagBits::eDepth);


}

void Core::destroySwapChain(){
    for (auto frame : _swapChainContext._frames) {
        destroyImageView(frame._view);
        _device->destroyFence(frame._inFlight);
        _device->destroySemaphore(frame._imageAvailable);
        _device->destroySemaphore(frame._renderFinished);
    }
    destroyImageView(_swapChainDepthImageView);

    _device->destroySwapchainKHR(_swapChainContext._swapChain);
    destroyImage(_swapChainDepthImage);
}


vk::ShaderModule Core::createShaderModule(const std::vector<uint32_t> code) {
    vk::ShaderModuleCreateInfo createInfo({}, code);
    return _device->createShaderModule(createInfo);
}

vk::ShaderModule Core::loadShaderModule(std::string src) {

    auto fileExtension = src.substr(src.find_last_of('.'));

    shaderc_shader_kind stage;
    if (fileExtension == ".vert"){
        stage = shaderc_glsl_vertex_shader;
    }
    else if (fileExtension == ".frag"){
        stage = shaderc_glsl_fragment_shader;
    }
    else if (fileExtension == ".comp"){
        stage = shaderc_glsl_compute_shader;
    }
    else if (fileExtension == ".geom"){
        stage = shaderc_glsl_geometry_shader;
    }

    std::ifstream input_file(src);
    if (!input_file.is_open()) {
        std::cerr << "Could not open the file - '" << src << "'" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string shaderCodeGlsl = std::string((std::istreambuf_iterator<char>(input_file)), std::istreambuf_iterator<char>());

    //  auto preprocessed = preprocess_shader("shader_src", stage, kShaderSource);
    //  std::cout << "Compiled a vertex shader resulting in preprocessed text:" << std::endl  << preprocessed << std::endl;

    std::cout << "Compiling shader  " << src << "" << std::endl;
    auto spirv = compile_file("shader_src", stage, shaderCodeGlsl.c_str()); //, shaderc_optimization_level_performance

    return createShaderModule(spirv);
}

vk::Format Core::findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
    for (vk::Format format : candidates) {
        vk::FormatProperties props;
        _physicalDevice.getFormatProperties(format, &props);

        if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
            return format;
        } else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
            return format;
        }
    }

    throw std::runtime_error("failed to find supported format!");
}

vk::Format Core::findDepthFormat() {
    return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment
    );
}

vk::Result gpu::Core::acquireNextImageKHR(uint32_t* imageIndex, vk::Semaphore semaphore, vk::Fence fence)
{
    vk::Result result;

    try{
        result = _device->acquireNextImageKHR(getSwapChain(), UINT64_MAX, semaphore, fence, imageIndex);
    }
    catch(const vk::OutOfDateKHRError outOfDateError){
        result = vk::Result::eErrorOutOfDateKHR;
    }
    catch(const std::exception& e){
        std::cerr << e.what() << '\n';
    }
    return result;
}

vk::Result gpu::Core::presentKHR(uint32_t imageIndex, std::vector<vk::Semaphore> semaphores)
{
        std::vector<vk::SwapchainKHR> swapChains = { getSwapChain() };
        vk::PresentInfoKHR presentInfo(semaphores, swapChains, imageIndex);
        vk::Result result;
        try{
            result = presentQueue.presentKHR(presentInfo);
        }
        catch(const vk::OutOfDateKHRError outOfDateError){
            result = vk::Result::eErrorOutOfDateKHR;
        }
        catch(const std::exception& e){
            std::cerr << e.what() << '\n';
        }
        return result;
}

void gpu::Core::createComputeContext(ComputeContext& context)
{   
    vk::SemaphoreCreateInfo semaphoreInfo;
    vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);

    context._frames.resize(gpu::MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        context._frames[i]._computeFinished = _device->createSemaphore(semaphoreInfo);
        context._frames[i]._inFlight = _device->createFence(fenceInfo);
    }
}

void gpu::Core::destroyComputeContext(ComputeContext& context)
{
    for (auto frame : context._frames) {
        _device->destroySemaphore(frame._computeFinished);
        _device->destroyFence(frame._inFlight);
    }
}