#include "granular_matter.h"
#include <chrono>
#include "iostream"

GranularMatter::GranularMatter(gpu::Core* core)
{
    m_core = core;
    boundaryModule = m_core->loadShaderModule(SHADER_PATH"/boundary.comp");
    densityPressureModule = m_core->loadShaderModule(SHADER_PATH"/density_pressure.comp");
    forceModule = m_core->loadShaderModule(SHADER_PATH"/force.comp");
    integrateModule = m_core->loadShaderModule(SHADER_PATH"/integrate.comp");
    
    particles = std::vector<Particle>(computeSpace.x * computeSpace.y * computeSpace.z);
    std::generate(particles.begin(), particles.end(), [this]() {
        return Particle((static_cast<float>(std::rand()) / RAND_MAX) * settings.DOMAIN_WIDTH, (static_cast<float>(std::rand()) / RAND_MAX) * settings.DOMAIN_HEIGHT);
    });
    // equilibrium distance
    float r0 = 0.5f * settings.kernelRadius;
    
    for(float i = 0.f;i < settings.DOMAIN_HEIGHT; i+=r0){
        boundaryParticles.push_back(Particle(0.f, i));
        boundaryParticles.push_back(Particle(settings.DOMAIN_WIDTH, i));
    }

    for(float i = 0.f;i < settings.DOMAIN_WIDTH; i+=r0){
        boundaryParticles.push_back(Particle(i, 0.f));
        boundaryParticles.push_back(Particle(i, settings.DOMAIN_HEIGHT));
    }
    

    particlesBufferA.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    particlesBufferB.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    settingsBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    boundaryParticlesBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        particlesBufferA[i] = m_core->bufferFromData(particles.data(),sizeof(Particle) * particles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        particlesBufferB[i] = m_core->bufferFromData(particles.data(),sizeof(Particle) * particles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        settingsBuffer[i]   = m_core->bufferFromData(&settings, sizeof(SPHSettings), vk::BufferUsageFlagBits::eUniformBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU);
        boundaryParticlesBuffer[i] = m_core->bufferFromData(boundaryParticles.data(),sizeof(Particle) * boundaryParticles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
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
    commandBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
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
    //* Boundary
    device.destroyShaderModule(boundaryModule);
    device.destroyPipelineLayout(boundaryLayout);
    device.destroyPipeline(boundaryPipeline);

    //* destroy density pressure stuff
    device.destroyPipelineLayout(densityPressureLayout);
    device.destroyShaderModule(densityPressureModule);
    device.destroyPipeline(densityPressurePipeline);

    //* destroy force stuff
    device.destroyPipelineLayout(forceLayout);
    device.destroyShaderModule(forceModule);
    device.destroyPipeline(forcePipeline);

    //* destroy integrate stuff
    device.destroyPipelineLayout(integrateLayout);
    device.destroyShaderModule(integrateModule);
    device.destroyPipeline(integratePipeline);

    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        m_core->destroyBuffer(particlesBufferA[i]);
        m_core->destroyBuffer(particlesBufferB[i]);
        m_core->destroyBuffer(settingsBuffer[i]);
        m_core->destroyBuffer(boundaryParticlesBuffer[i]);
    }
    
    device.destroyDescriptorSetLayout(descriptorSetLayout);
    device.destroyDescriptorPool(descriptorPool);
}
void GranularMatter::initFrameResources(){
    createCommandBuffers();
}

std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();

void GranularMatter::update(int currentFrame, int imageIndex){
    auto currentTime = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    startTime = std::chrono::high_resolution_clock::now();
   
    settings.dt = 0.1f * dt; //0.02f *
    // std::cout << dt << std::endl;
    void* mappedData = m_core->mapBuffer(settingsBuffer[currentFrame]);
    memcpy(mappedData, &settings, (size_t) sizeof(SPHSettings));
    m_core->flushBuffer(settingsBuffer[currentFrame], 0, (size_t) sizeof(SPHSettings));
    m_core->unmapBuffer(settingsBuffer[currentFrame]);

    vk::CommandBufferBeginInfo beginInfo;
    try{
        commandBuffers[currentFrame].begin(beginInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
    //* Copy data from last 
    // Todo: replace with alternating buffers
    vk::BufferCopy copyRegion(0, 0, sizeof(Particle) * particles.size());
    commandBuffers[currentFrame].copyBuffer(particlesBufferB[(currentFrame - 1) % gpu::MAX_FRAMES_IN_FLIGHT], particlesBufferA[currentFrame], 1, &copyRegion);
    //* Wait for copy action
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);
    // Todo: Find neighbors
    // Todo: Add init pass
    // Todo: do 3 times
        // Todo: add predict velocity/position pass
        // Todo: add predict pressure/density pass
        // Todo: add force pass
    // Todo: add udate velocity/position
    //* compute boundary densities
    commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, boundaryPipeline);
    commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, boundaryLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
    commandBuffers[currentFrame].dispatch(boundaryParticles.size(), 1, 1);
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);
    //* compute pressure
    commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, densityPressurePipeline);
    commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, densityPressureLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
    commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);
    //* compute forces
    commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, forcePipeline);
    commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, forceLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
    commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);
    //* integrate
    commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, integratePipeline);
    commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, integrateLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
    commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
    //* submit calls

    try{
        commandBuffers[currentFrame].end();
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}

void GranularMatter::createDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding particlesLayoutBindingIn(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
    vk::DescriptorSetLayoutBinding particlesLayoutBindingOut(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
    vk::DescriptorSetLayoutBinding settingsLayoutBinding(2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute | vk::ShaderStageFlagBits::eVertex, nullptr);
    vk::DescriptorSetLayoutBinding boundaryLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);

    std::array<vk::DescriptorSetLayoutBinding, 4> bindings = {
        particlesLayoutBindingIn, 
        particlesLayoutBindingOut,
        settingsLayoutBinding,
        boundaryLayoutBinding
    };
    vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);

    try{
        descriptorSetLayout = m_core->getDevice().createDescriptorSetLayout(layoutInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}
// Todo: connect all simulation frames
void GranularMatter::createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(gpu::MAX_FRAMES_IN_FLIGHT, descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo(descriptorPool, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT), layouts.data());
    descriptorSets.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    try{
        descriptorSets = m_core->getDevice().allocateDescriptorSets(allocInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
    
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {

        vk::DescriptorBufferInfo bufferInfoA(particlesBufferA[i], 0, sizeof(Particle) * particles.size());
        vk::DescriptorBufferInfo bufferInfoB(particlesBufferB[i], 0, sizeof(Particle) * particles.size());
        vk::DescriptorBufferInfo bufferInfoSettings(settingsBuffer[i], 0, sizeof(SPHSettings));
        vk::DescriptorBufferInfo bufferInfoBoundary(boundaryParticlesBuffer[i], 0, sizeof(Particle) * boundaryParticles.size());
    
        vk::WriteDescriptorSet descriptorWriteA(descriptorSets[i], 0, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &bufferInfoA);
        vk::WriteDescriptorSet descriptorWriteB(descriptorSets[i], 1, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &bufferInfoB);
        vk::WriteDescriptorSet descriptorWriteSettings(descriptorSets[i], 2, 0, 1, vk::DescriptorType::eUniformBuffer, {}, &bufferInfoSettings);
        vk::WriteDescriptorSet descriptorWriteBoundary(descriptorSets[i], 3, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &bufferInfoBoundary);

        std::array<vk::WriteDescriptorSet, 4> descriptorWrites{
            descriptorWriteA, 
            descriptorWriteB,
            descriptorWriteSettings,
            descriptorWriteBoundary
        };
        
        m_core->getDevice().updateDescriptorSets(descriptorWrites, nullptr);
    }
}

void GranularMatter::createDescriptorPool() {
    vk::DescriptorPoolSize particlesInSize(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
    vk::DescriptorPoolSize particlesOutSize(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
    vk::DescriptorPoolSize settingsSize(vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
    std::array<vk::DescriptorPoolSize, 3> poolSizes{
        particlesInSize, 
        particlesOutSize,
        settingsSize
    };

    vk::DescriptorPoolCreateInfo poolInfo({}, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT), poolSizes);
    try{
        descriptorPool = m_core->getDevice().createDescriptorPool(poolInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
}
void GranularMatter::createComputePipeline(){
    vk::Result result;
    //* Boundary Pipeline
    vk::PipelineShaderStageCreateInfo boundaryStageInfo{
        {},
        vk::ShaderStageFlagBits::eCompute,
        boundaryModule,
        "main"
    };

    vk::PipelineLayoutCreateInfo boundaryLayoutInfo{
        {},
        descriptorSetLayout,
        {}
    };

    try{
        boundaryLayout = m_core->getDevice().createPipelineLayout(boundaryLayoutInfo, nullptr);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }

    vk::ComputePipelineCreateInfo boundaryPipelineInfo{
        {},
        boundaryStageInfo,
        boundaryLayout,
    };

    
    std::tie(result, boundaryPipeline) = m_core->getDevice().createComputePipeline( nullptr, boundaryPipelineInfo);
    switch ( result ){
        case vk::Result::eSuccess: break;
        default: throw std::runtime_error("failed to create compute Pipeline!");
    }
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
    //* Force Pipeline
    vk::PipelineShaderStageCreateInfo forceStageInfo{
        {},
        vk::ShaderStageFlagBits::eCompute,
        forceModule,
        "main"
    };

    vk::PipelineLayoutCreateInfo forceLayoutInfo{
        {},
        descriptorSetLayout,
        {}
    };

    try{
        forceLayout = m_core->getDevice().createPipelineLayout(forceLayoutInfo, nullptr);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }

    vk::ComputePipelineCreateInfo forcePipelineInfo{
        {},
        forceStageInfo,
        forceLayout,
    };

    
    std::tie(result, forcePipeline) = m_core->getDevice().createComputePipeline( nullptr, forcePipelineInfo);
    switch ( result ){
        case vk::Result::eSuccess: break;
        default: throw std::runtime_error("failed to create compute Pipeline!");
    }
    //* Integrate Pipeline
    vk::PipelineShaderStageCreateInfo integrateStageInfo{
        {},
        vk::ShaderStageFlagBits::eCompute,
        integrateModule,
        "main"
    };

    vk::PipelineLayoutCreateInfo integrateLayoutInfo{
        {},
        descriptorSetLayout,
        {}
    };

    try{
        integrateLayout = m_core->getDevice().createPipelineLayout(integrateLayoutInfo, nullptr);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }

    vk::ComputePipelineCreateInfo integratePipelineInfo{
        {},
        integrateStageInfo,
        integrateLayout,
    };

    
    std::tie(result, integratePipeline) = m_core->getDevice().createComputePipeline( nullptr, integratePipelineInfo);
    switch ( result ){
        case vk::Result::eSuccess: break;
        default: throw std::runtime_error("failed to create compute Pipeline!");
    }
}