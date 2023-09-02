#include "granular_matter.h"

GranularMatter::GranularMatter(gpu::Core* core)
{
    m_core = core;
    densityPressureModule = m_core->loadShaderModule(SHADER_PATH"/density_pressure.comp");
    // forceModule = m_core->loadShaderModule(SHADER_PATH"/force.comp");
    // integrateModule = m_core->loadShaderModule(SHADER_PATH"/integrate.comp");
    
    particles = std::vector<Particle>(64 * 64);
    std::generate(particles.begin(), particles.end(), [this]() {
        return Particle((static_cast<float>(std::rand()) / RAND_MAX) * settings.DOMAIN_WIDTH, (static_cast<float>(std::rand()) / RAND_MAX) * settings.DOMAIN_HEIGHT);
    });

    particlesBufferA.resize(m_core->getSwapChainImageCount());
    particlesBufferB.resize(m_core->getSwapChainImageCount());
    settingsBuffer.resize(m_core->getSwapChainImageCount());
    for (size_t i = 0; i < m_core->getSwapChainImageCount(); i++) {
        particlesBufferA[i] = m_core->bufferFromData(particles.data(),particles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        particlesBufferB[i] = m_core->bufferFromData(particles.data(),particles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        settingsBuffer[i]   = m_core->bufferFromData(&settings, sizeof(SPHSettings), vk::BufferUsageFlagBits::eUniformBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
    }
    initFrameResources();
    createDescriptorPool();
    createDescriptorSetLayout();
    createDescriptorSets();
    createComputePipeline();
}

GranularMatter::~GranularMatter()
{
}
void GranularMatter::createCommandBuffers(){
    commandBuffers.resize(m_core->getSwapChainImageCount());
    vk::CommandBufferAllocateInfo allocInfo(m_core->getCommandPool(), vk::CommandBufferLevel::ePrimary, (uint32_t) commandBuffers.size());
    try{
        commandBuffers = m_core->getDevice().allocateCommandBuffers(allocInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}
void GranularMatter::destroyFrameResources(){
    vk::Device device = m_core->getDevice();
    device.freeCommandBuffers(m_core->getCommandPool(), commandBuffers);
}
void GranularMatter::destroy(){
    destroyFrameResources();
    vk::Device device = m_core->getDevice();
    //* destroy density pressure stuff
    device.destroyPipelineLayout(densityPressureLayout);
    device.destroyShaderModule(densityPressureModule);
    device.destroyPipeline(densityPressurePipeline);

    for (size_t i = 0; i < m_core->getSwapChainImageCount(); i++) {
        m_core->destroyBuffer(particlesBufferA[i]);
        m_core->destroyBuffer(particlesBufferB[i]);
        m_core->destroyBuffer(settingsBuffer[i]);
    }
    
    device.destroyDescriptorSetLayout(descriptorSetLayout);
}
void GranularMatter::initFrameResources(){
    createCommandBuffers();
}
void GranularMatter::update(int imageIndex){
    vk::CommandBufferBeginInfo beginInfo;
    try{
        commandBuffers[imageIndex].begin(beginInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
    commandBuffers[imageIndex].bindPipeline(vk::PipelineBindPoint::eCompute, densityPressurePipeline);
    commandBuffers[imageIndex].bindDescriptorSets(vk::PipelineBindPoint::eCompute, densityPressureLayout, 0, 1, &descriptorSets[imageIndex], 0, nullptr);
    commandBuffers[imageIndex].dispatch(64, 64, 1);
  
    try{
        commandBuffers[imageIndex].end();
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}

void GranularMatter::createDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding particlesLayoutBindingIn(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
    vk::DescriptorSetLayoutBinding particlesLayoutBindingOut(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
    vk::DescriptorSetLayoutBinding settingsLayoutBinding(2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eVertex, nullptr);

    std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {
        particlesLayoutBindingIn, 
        particlesLayoutBindingOut,
        settingsLayoutBinding
    };
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);

    try{
        descriptorSetLayout = m_core->getDevice().createDescriptorSetLayout(layoutInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}

void GranularMatter::createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(m_core->getSwapChainImageCount(), descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, static_cast<uint32_t>(m_core->getSwapChainImageCount()), layouts.data());
    descriptorSets.resize(m_core->getSwapChainImageCount());
    try{
        descriptorSets = m_core->getDevice().allocateDescriptorSets(allocInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
    
    for (size_t i = 0; i < m_core->getSwapChainImageCount(); i++) {

        vk::DescriptorBufferInfo bufferInfoA(particlesBufferA[i], 0, particles.size());
        vk::DescriptorBufferInfo bufferInfoB(particlesBufferB[i], 0, particles.size());
        vk::DescriptorBufferInfo bufferInfoSettings(settingsBuffer[i], 0, sizeof(SPHSettings));
    
        vk::WriteDescriptorSet descriptorWriteA(descriptorSets[i], 0, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &bufferInfoA);
        vk::WriteDescriptorSet descriptorWriteB(descriptorSets[i], 1, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &bufferInfoB);
        vk::WriteDescriptorSet descriptorWriteSettings(descriptorSets[i], 2, 0, 1, vk::DescriptorType::eUniformBuffer, {}, &bufferInfoSettings);

        std::array<vk::WriteDescriptorSet, 3> descriptorWrites{
            descriptorWriteA, 
            descriptorWriteB,
            descriptorWriteSettings
        };
        
        m_core->getDevice().updateDescriptorSets(descriptorWrites, nullptr);
    }
}

void GranularMatter::createDescriptorPool() {
    vk::DescriptorPoolSize particlesInSize(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(m_core->getSwapChainImageCount()));
    vk::DescriptorPoolSize particlesOutSize(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(m_core->getSwapChainImageCount()));
    vk::DescriptorPoolSize settingsSize(vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(m_core->getSwapChainImageCount()));
    std::array<vk::DescriptorPoolSize, 3> poolSizes{
        particlesInSize, 
        particlesOutSize,
        settingsSize
    };

    vk::DescriptorPoolCreateInfo poolInfo({}, static_cast<uint32_t>(m_core->getSwapChainImageCount()), poolSizes);
    try{
        descriptorPool = m_core->getDevice().createDescriptorPool(poolInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}
void GranularMatter::createComputePipeline(){
    vk::Result result;
    //* Density Pressure Pipeline
    vk::PipelineShaderStageCreateInfo densityPressureStageInfo{
        {},
        vk::ShaderStageFlagBits::eCompute,
        densityPressureModule,
        "main"
    };

    vk::PipelineLayoutCreateInfo densityPressureLayoutInfo{
        {},
        descriptorSetLayout,
        {}
    };

    try{
        densityPressureLayout = m_core->getDevice().createPipelineLayout(densityPressureLayoutInfo, nullptr);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }

    vk::ComputePipelineCreateInfo densityPressurePipelineInfo{
        {},
        densityPressureStageInfo,
        densityPressureLayout,
    };

    
    std::tie(result, densityPressurePipeline) = m_core->getDevice().createComputePipeline( nullptr, densityPressurePipelineInfo);
    switch ( result ){
        case vk::Result::eSuccess: break;
        default: throw std::runtime_error("failed to create compute Pipeline!");
    }
    // //* Force Pipeline
    // vk::PipelineShaderStageCreateInfo forceStageInfo{
    //     {},
    //     vk::ShaderStageFlagBits::eCompute,
    //     forceModule,
    //     "main"
    // };

    // vk::PipelineLayoutCreateInfo forceLayoutInfo{
    //     {},
    //     descriptorSetLayout,
    //     {}
    // };

    // try{
    //     forceLayout = m_core->getDevice().createPipelineLayout(forceLayoutInfo, nullptr);
    // }catch(std::exception& e) {
    //     std::cerr << "Exception Thrown: " << e.what();
    // }

    // vk::ComputePipelineCreateInfo forcePipelineInfo{
    //     {},
    //     forceStageInfo,
    //     forceLayout,
    // };

    
    // std::tie(result, forcePipeline) = m_core->getDevice().createComputePipeline( nullptr, forcePipelineInfo);
    // switch ( result ){
    //     case vk::Result::eSuccess: break;
    //     default: throw std::runtime_error("failed to create compute Pipeline!");
    // }
    // //* Integrate Pipeline
    // vk::PipelineShaderStageCreateInfo integrateStageInfo{
    //     {},
    //     vk::ShaderStageFlagBits::eCompute,
    //     integrateModule,
    //     "main"
    // };

    // vk::PipelineLayoutCreateInfo integrateLayoutInfo{
    //     {},
    //     descriptorSetLayout,
    //     {}
    // };

    // try{
    //     integrateLayout = m_core->getDevice().createPipelineLayout(integrateLayoutInfo, nullptr);
    // }catch(std::exception& e) {
    //     std::cerr << "Exception Thrown: " << e.what();
    // }

    // vk::ComputePipelineCreateInfo integratePipelineInfo{
    //     {},
    //     integrateStageInfo,
    //     integrateLayout,
    // };

    
    // std::tie(result, integratePipeline) = m_core->getDevice().createComputePipeline( nullptr, integratePipelineInfo);
    // switch ( result ){
    //     case vk::Result::eSuccess: break;
    //     default: throw std::runtime_error("failed to create compute Pipeline!");
    // }
}