#include "core.h"

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

    vk::PhysicalDeviceFeatures deviceFeatures;
    deviceFeatures.samplerAnisotropy = VK_TRUE;

    //MacOS portability extension
    std::vector<vk::ExtensionProperties> extensionProperties =  physicalDevice.enumerateDeviceExtensionProperties();
    for(auto extensionProperty : extensionProperties){
        if(std::string(extensionProperty.extensionName.data()) == std::string(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME))
            deviceExtensions.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }

    vk::DeviceCreateInfo createInfo;
    vk::PhysicalDeviceHostQueryResetFeatures resetFeatures{};
    if (m_enableValidation) {
        createInfo = vk::DeviceCreateInfo({}, queueCreateInfos, validationLayers, deviceExtensions, &deviceFeatures, &resetFeatures);
    }
    else{
        createInfo = vk::DeviceCreateInfo({}, queueCreateInfos, {}, deviceExtensions, &deviceFeatures, &resetFeatures);
    }
    try{
        device = physicalDevice.createDevice(createInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
    graphicsQueue = device.getQueue(indices.graphicsFamily.value(), 0);
    computeQueue = device.getQueue(indices.computeFamily.value(), 0);
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
vk::Buffer Core::bufferFromData(void* data, size_t size, vk::BufferUsageFlags bufferUsage, VmaMemoryUsage memoryUsage){
    if(memoryUsage != VMA_MEMORY_USAGE_CPU_ONLY && 
        memoryUsage != VMA_MEMORY_USAGE_GPU_ONLY &&
        memoryUsage != VMA_MEMORY_USAGE_CPU_TO_GPU){
        throw std::exception("Unsupported memory usage.");
    }

    vk::Buffer stagingBuffer;

    if(memoryUsage == VMA_MEMORY_USAGE_GPU_ONLY){
        stagingBuffer = createBuffer(size, vk::BufferUsageFlagBits::eTransferSrc, VMA_MEMORY_USAGE_CPU_ONLY);
    }
    else if(memoryUsage == VMA_MEMORY_USAGE_CPU_TO_GPU){
        stagingBuffer = createBuffer(size, bufferUsage, memoryUsage);
    }
    
    void* mappedData = mapBuffer(stagingBuffer);
    memcpy(mappedData, data, (size_t) size);
    if(memoryUsage == VMA_MEMORY_USAGE_CPU_TO_GPU){
        flushBuffer(stagingBuffer, 0, size);
    }
    unmapBuffer(stagingBuffer);

    if(memoryUsage == VMA_MEMORY_USAGE_CPU_ONLY || memoryUsage == VMA_MEMORY_USAGE_CPU_TO_GPU){
        return stagingBuffer;
    }

    vk::Buffer buffer = createBuffer(size, vk::BufferUsageFlagBits::eTransferDst | bufferUsage, VMA_MEMORY_USAGE_GPU_ONLY);
    copyBufferToBuffer(stagingBuffer, buffer, size);
    destroyBuffer(stagingBuffer);

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

std::vector<vk::DescriptorSet> gpu::Core::allocateDescriptorSets(vk::DescriptorSetLayout layout, vk::DescriptorPool pool, uint32_t count)
{
    std::vector<vk::DescriptorSetLayout> layouts(count, layout);
    vk::DescriptorSetAllocateInfo allocInfo(pool, count, layouts.data());
    try{
        return device.allocateDescriptorSets(allocInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}

void gpu::Core::updateDescriptorSet(vk::DescriptorSet set, std::vector<gpu::DescriptorWrite> writes)
{
    std::vector<vk::WriteDescriptorSet> descriptorWrites;

    for (auto &w : writes){
        vk::DescriptorBufferInfo* bufferInfo = new vk::DescriptorBufferInfo(w.buffer, 0, w.size);
        vk::WriteDescriptorSet descriptorWrite(set, w.binding, 0, 1, w.type, {}, bufferInfo);
        descriptorWrites.push_back(descriptorWrite);
    }

    device.updateDescriptorSets(descriptorWrites, nullptr);
}

vk::DescriptorSetLayout gpu::Core::createDescriptorSetLayout(std::vector<DescriptorSetBinding> bindings)
{
    std::vector<vk::DescriptorSetLayoutBinding> descriptorSetLayoutBindings;
    for(auto b : bindings){
        descriptorSetLayoutBindings.push_back(vk::DescriptorSetLayoutBinding(b.binding, b.type, 1, b.stages, nullptr));
    }
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, descriptorSetLayoutBindings);
    try{
        return device.createDescriptorSetLayout(layoutInfo);
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
vk::Image Core::createImage(vk::ImageUsageFlags imageUsage, VmaMemoryUsage memoryUsage, uint32_t width, uint32_t height, vk::Format format, vk::ImageTiling tiling) {
    vk::Image image;
    VmaAllocation imageAllocation;
    vk::ImageCreateInfo imageInfo({}, vk::ImageType::e2D, format, vk::Extent3D{{width, height}, 1}, 1, 1, vk::SampleCountFlagBits::e1, tiling, imageUsage, vk::SharingMode::eExclusive, {}, {}, vk::ImageLayout::eUndefined);
    VmaAllocationCreateInfo allocInfoImage = {};
    allocInfoImage.usage = memoryUsage;
    if(vmaCreateImage(allocator, reinterpret_cast<VkImageCreateInfo*>(&imageInfo), &allocInfoImage, reinterpret_cast<VkImage*>(&image), &imageAllocation, nullptr) != VK_SUCCESS){
        throw std::runtime_error("failed to create image!");
    }
    m_imageAllocations[image] = imageAllocation;
    return image;
}
void Core::destroyImage(vk::Image image){
    vmaDestroyImage(allocator, image, m_imageAllocations[image]);
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

vk::Sampler Core::createTextureSampler() {
    vk::Sampler sampler;
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
        if (availableFormat.format == vk::Format::eR8G8B8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
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