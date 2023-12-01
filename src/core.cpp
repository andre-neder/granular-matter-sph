#include "core.h"

#define MAX_VARIABLE_DESCRIPTOR_COUNT 32

using namespace gpu;

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
    createSwapChain(window);
    createSwapChainImageViews();
}

void Core::destroy(){
    device.destroyCommandPool(commandPool);

    allocator.destroy();
    device.destroy();
    instance.destroySurfaceKHR(surface);
    if (m_enableValidation) {
        instance.destroyDebugUtilsMessengerEXT(m_debugMessenger);
    }
    instance.destroy();
}

uint32_t gpu::Core::getIdealWorkGroupSize()
{
    uint32_t vendorID = physicalDevice.getProperties().vendorID;
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

    auto m_deviceFeatures2 = pDevice.getFeatures2<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceBufferDeviceAddressFeatures, vk::PhysicalDeviceDescriptorIndexingFeatures, vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT>();
    bool supportsAllFeatures =
        m_deviceFeatures2.get<vk::PhysicalDeviceFeatures2>().features.samplerAnisotropy &&
        m_deviceFeatures2.get<vk::PhysicalDeviceFeatures2>().features.geometryShader &&
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

    //MacOS portability extension
    std::vector<vk::ExtensionProperties> extensionProperties =  physicalDevice.enumerateDeviceExtensionProperties();
    for(auto extensionProperty : extensionProperties){
        if(std::string(extensionProperty.extensionName.data()) == std::string(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
            deviceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }

    vk::DeviceCreateInfo deviceCreateInfo;
	if (m_enableValidation)
	{
		deviceCreateInfo = vk::DeviceCreateInfo({}, queueCreateInfos, validationLayers, deviceExtensions, {});
	}
	else
	{
		deviceCreateInfo = vk::DeviceCreateInfo({}, queueCreateInfos, {}, deviceExtensions, {});
	}
    vk::StructureChain<vk::DeviceCreateInfo, vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceRayTracingPipelineFeaturesKHR, vk::PhysicalDeviceAccelerationStructureFeaturesKHR, vk::PhysicalDeviceBufferDeviceAddressFeatures, vk::PhysicalDeviceDescriptorIndexingFeatures, vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT> deviceFeatureCreateInfo = {
		deviceCreateInfo,
		vk::PhysicalDeviceFeatures2().setFeatures(vk::PhysicalDeviceFeatures().setSamplerAnisotropy(true).setGeometryShader(true).setShaderSampledImageArrayDynamicIndexing(true)),
		vk::PhysicalDeviceRayTracingPipelineFeaturesKHR().setRayTracingPipeline(true),
		vk::PhysicalDeviceAccelerationStructureFeaturesKHR().setAccelerationStructure(true),
		vk::PhysicalDeviceBufferDeviceAddressFeatures().setBufferDeviceAddress(true),
		vk::PhysicalDeviceDescriptorIndexingFeatures().setRuntimeDescriptorArray(true).setShaderSampledImageArrayNonUniformIndexing(true).setDescriptorBindingVariableDescriptorCount(true).setDescriptorBindingPartiallyBound(true),
        vk::PhysicalDeviceShaderAtomicFloatFeaturesEXT().setShaderBufferFloat32Atomics(true).setShaderBufferFloat32AtomicAdd(true)
	};
    
    try
    {
        device = physicalDevice.createDevice(deviceFeatureCreateInfo.get<vk::DeviceCreateInfo>());
    }
    catch(std::exception& e) 
    {
        std::cerr << "Exception Thrown: " << e.what();
    }
    graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
    computeQueue = device.getQueue(indices.computeFamily.value(), 0);
    presentQueue = device.getQueue(indices.presentFamily.value(), 0);
}

void Core::createAllocator(){
    vma::AllocatorCreateInfo allocatorInfo;
    allocatorInfo.flags = vma::AllocatorCreateFlagBits::eBufferDeviceAddress;
    allocatorInfo.physicalDevice = physicalDevice;
    allocatorInfo.device = device;
    allocatorInfo.instance = instance;
    allocatorInfo.vulkanApiVersion = VK_API_VERSION_1_2;
	try
	{
		allocator = vma::createAllocator(allocatorInfo);
	}
	catch (std::exception &e)
	{
		std::cerr << "Exception Thrown: " << e.what();
	}
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

    try
    {
        std::tie(buffer, allocation) = allocator.createBuffer(bufferInfo, bufferAllocInfo);
    }
    catch (std::exception &e)
    {
        std::cerr << "Exception Thrown: " << e.what();
    }
    m_bufferAllocations[buffer] = allocation;
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
void* Core::mapBuffer(vk::Buffer buffer){
    void* mappedData = allocator.mapMemory(m_bufferAllocations[buffer]);
    return mappedData;
}
void Core::unmapBuffer(vk::Buffer buffer){
    allocator.unmapMemory(m_bufferAllocations[buffer]);
}
void Core::destroyBuffer(vk::Buffer buffer){
    allocator.destroyBuffer(buffer, m_bufferAllocations[buffer]);
    m_bufferAllocations.erase(buffer);
}
void Core::flushBuffer(vk::Buffer buffer, size_t offset, size_t size){
    allocator.flushAllocation(m_bufferAllocations[buffer], offset, size);
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

void gpu::Core::beginCommands(vk::CommandBuffer commandBuffer, vk::CommandBufferBeginInfo beginInfo)
{
    try{
        commandBuffer.begin(beginInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}

void gpu::Core::endCommands(vk::CommandBuffer commandBuffer)
{
    try{
        commandBuffer.end();
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}

void gpu::Core::createTimestampQueryPool(vk::QueryPool* pool)
{
    vk::QueryPoolCreateInfo createInfo{
        {},
        vk::QueryType::eTimestamp,
        MAX_QUERY_POOL_COUNT, 
        {}
    };
    
    vk::Result result = device.createQueryPool(&createInfo, nullptr, pool);
    if (result != vk::Result::eSuccess)
    {
        throw std::runtime_error("Failed to create time query pool!");
    }

}


std::vector<uint64_t> gpu::Core::getTimestampQueryPoolResults(vk::QueryPool *pool)
{
    uint64_t buffer[MAX_QUERY_POOL_COUNT];

    vk::Result result = device.getQueryPoolResults(*pool, 0, MAX_QUERY_POOL_COUNT, sizeof(uint64_t) * MAX_QUERY_POOL_COUNT, buffer, sizeof(uint64_t), vk::QueryResultFlagBits::e64);
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
    // if the descriptor set layout has the flag
    // auto flags = m_descriptorBindingFlags.at(layout);
    // if((flags & (vk::DescriptorBindingFlagBits::eVariableDescriptorCount | vk::DescriptorBindingFlagBits::ePartiallyBound)) == (vk::DescriptorBindingFlagBits::eVariableDescriptorCount | vk::DescriptorBindingFlagBits::ePartiallyBound)){
        auto descriptorCount = m_descriptorCount.at(layout);
        std::vector<uint32_t> counts(count, descriptorCount);
        vk::DescriptorSetVariableDescriptorCountAllocateInfo variableDescriptorCountAllocInfo{count, counts.data()};
        allocInfo.pNext = &variableDescriptorCountAllocInfo;
    // }
    try{
        auto sets = device.allocateDescriptorSets(allocInfo);
        for(auto& set : sets){
            //* create a write queue for each set
            m_descriptorWrites[set] = std::vector<vk::WriteDescriptorSet>();
        }
        return sets;
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}

void gpu::Core::updateDescriptorSet(vk::DescriptorSet set)
{
    auto& descriptorWrites = m_descriptorWrites.at(set);
    device.updateDescriptorSets(descriptorWrites, nullptr);
    descriptorWrites.clear();
}


void Core::addDescriptorWrite(vk::DescriptorSet set, gpu::BufferDescriptorWrite write)
{
    //Todo: Cleanup
    vk::DescriptorBufferInfo* bufferInfo = new vk::DescriptorBufferInfo(write.buffer, 0, write.size);
    vk::WriteDescriptorSet descriptorWrite(set, write.binding, 0, 1, write.type, {}, bufferInfo);
    m_descriptorWrites.at(set).push_back(descriptorWrite);
}

void Core::addDescriptorWrite(vk::DescriptorSet set, gpu::ImageDescriptorWrite write)
{
    //Todo: Cleanup
    std::vector<vk::DescriptorImageInfo>* imageInfos = new std::vector<vk::DescriptorImageInfo>(); 
    if(write.imageViews.size() == 0){
        imageInfos->push_back(vk::DescriptorImageInfo(write.sampler, {}, {}));
    }
    else{
        for(auto view : write.imageViews){
            imageInfos->push_back(vk::DescriptorImageInfo(write.sampler, view, write.imageLayout));
        }
        
    }
    vk::WriteDescriptorSet descriptorWrite(set, write.binding, 0, (uint32_t)imageInfos->size(), write.type, imageInfos->data());
    m_descriptorWrites.at(set).push_back(descriptorWrite);
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
    

    try{
        auto descriptorSetLayout = device.createDescriptorSetLayout(layoutInfo);
        m_descriptorCount[descriptorSetLayout] = descriptorCount;
        m_descriptorBindingFlags[descriptorSetLayout] = layoutFlags;
        return descriptorSetLayout;
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}

vk::DescriptorPool gpu::Core::createDescriptorPool(std::vector<vk::DescriptorPoolSize> sizes, uint32_t maxSets)
{
    vk::DescriptorPoolCreateInfo poolInfo({}, maxSets, sizes);
    try{
        return device.createDescriptorPool(poolInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}

void gpu::Core::destroyDescriptorPool(vk::DescriptorPool pool)
{
    device.destroyDescriptorPool(pool);
}

void gpu::Core::destroyDescriptorSetLayout(vk::DescriptorSetLayout layout)
{
    device.destroyDescriptorSetLayout(layout);
}

void Core::createCommandPool() {
    QueueFamilyIndices queueFamilyIndices = findQueueFamilies(physicalDevice);
    vk::CommandPoolCreateInfo poolInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, queueFamilyIndices.graphicsFamily.value());
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
    try{
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
    }
    catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }

    vk::ImageMemoryBarrier barrier(srcAccessMask, dstAccessMask, oldLayout, newLayout, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image, vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));

    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {}, {}, {}, 1, &barrier);

    endSingleTimeCommands(commandBuffer);
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

    try
    {
        std::tie(image, allocation) = allocator.createImage(imageInfo, imageAllocInfo);
    }
    catch (std::exception &e)
    {
        std::cerr << "Exception Thrown: " << e.what();
    }
    m_imageAllocations[image] = allocation;
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

    try
    {
        std::tie(image, allocation) = allocator.createImage(imageInfo, imageAllocInfo);
    }
    catch (std::exception &e)
    {
        std::cerr << "Exception Thrown: " << e.what();
    }
    m_imageAllocations[image] = allocation;
    return image;
}

void Core::destroyImage(vk::Image image){
    allocator.destroyImage(image, m_imageAllocations[image]);
    m_imageAllocations.erase(image);
}

void Core::destroyImageView(vk::ImageView view){
    device.destroyImageView(view);
}

vk::ImageView Core::createImageView(vk::Image image, vk::Format format) {
    vk::ImageViewCreateInfo viewInfo({}, image, vk::ImageViewType::e2D, format, {}, vk::ImageSubresourceRange( vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
    vk::ImageView imageView;
    try{
        imageView = device.createImageView(viewInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
    return imageView;
}

vk::Sampler Core::createSampler(vk::SamplerAddressMode addressMode, vk::BorderColor borderColor, vk::Bool32 enableAnisotropy) {
    vk::Sampler sampler;
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();

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
    try{
        sampler = device.createSampler(samplerInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
    return sampler;
}
void Core::destroySampler(vk::Sampler sampler){
    device.destroySampler(sampler);
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
    SwapChainSupportDetails swapChainSupport = querySwapChainSupport(physicalDevice);

    surfaceFormat = chooseSwapSurfaceFormat(swapChainSupport.formats);
    vk::PresentModeKHR presentMode = chooseSwapPresentMode(swapChainSupport.presentModes);
    vk::Extent2D extent = chooseSwapExtent(swapChainSupport.capabilities, window);

    uint32_t imageCount = swapChainSupport.capabilities.minImageCount + 1;
    if (swapChainSupport.capabilities.maxImageCount > 0 && imageCount > swapChainSupport.capabilities.maxImageCount) {
        imageCount = swapChainSupport.capabilities.maxImageCount;
    }

    QueueFamilyIndices indices = findQueueFamilies(physicalDevice);
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
void Core::createSwapChainImageViews(){
    swapChainImageViews.resize(getSwapChainImageCount());
    for (size_t i = 0; i < getSwapChainImageCount(); i++) {
        swapChainImageViews[i] = createImageView(swapChainImages[i], swapChainImageFormat);
    }
}
void Core::destroySwapChainImageViews(){
    for (auto imageView : swapChainImageViews) {
        destroyImageView(imageView);
    }
}
void Core::destroySwapChain(){
    device.destroySwapchainKHR(swapChain);
}


vk::ShaderModule Core::createShaderModule(const std::vector<uint32_t> code) {
    vk::ShaderModuleCreateInfo createInfo({}, code);
    vk::ShaderModule shaderModule;
    try{
        shaderModule = device.createShaderModule(createInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
    return shaderModule;
}

vk::ShaderModule Core::loadShaderModule(std::string src) {
    vk::ShaderModule shaderModule;
    glslang::InitializeProcess();
    std::vector<uint32_t> shaderCodeSPIRV;
    std::string fileExtension = src.substr(src.find_last_of('.'));
    vk::ShaderStageFlagBits stage;
    if (fileExtension == ".vert"){
        stage = vk::ShaderStageFlagBits::eVertex;
    }
    else if (fileExtension == ".frag"){
        stage = vk::ShaderStageFlagBits::eFragment;
    }
    else if (fileExtension == ".comp"){
        stage = vk::ShaderStageFlagBits::eCompute;
    }
    else if (fileExtension == ".geom"){
        stage = vk::ShaderStageFlagBits::eGeometry;
    }
    SpirvHelper::GLSLtoSPV(stage, src, shaderCodeSPIRV);
    try{
        shaderModule = createShaderModule(shaderCodeSPIRV);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
    glslang::FinalizeProcess();
    return shaderModule;
}