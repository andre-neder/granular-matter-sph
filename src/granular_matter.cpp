#include "granular_matter.h"
#include <chrono>
#include "iostream"
#include "global.h"


GranularMatter::GranularMatter(gpu::Core* core)
{
    m_core = core;
    
    // particles = std::vector<Particle>(computeSpace.x * computeSpace.y * computeSpace.z);
    // std::generate(particles.begin(), particles.end(), [this]() {
    //     return Particle(((static_cast<float>(std::rand()) / RAND_MAX) / 4.f + 0.375f) * settings.DOMAIN_WIDTH, ((static_cast<float>(std::rand()) / RAND_MAX  / 2.f )) * settings.DOMAIN_HEIGHT);
    // });
    // equilibrium distance
    float r0 = 0.5f * settings.kernelRadius;

    float initialDistance = 0.5 * settings.kernelRadius;

    for(int i = 0;i < computeSpace.x ; i++){
        for(int j = 0;j < computeSpace.y ; j++){
            particles.push_back(Particle(i * initialDistance + 0.5 * settings.DOMAIN_WIDTH  ,j * initialDistance + 0.5));
        }
    }
    
    for(float i = 0.f;i < settings.DOMAIN_HEIGHT; i+=r0){
        boundaryParticles.push_back(BoundaryParticle(0.f, i, 1.f, 0.f));
        boundaryParticles.push_back(BoundaryParticle(settings.DOMAIN_WIDTH, i, -1.f, 0.f));
    }

    for(float i = 0.f;i < settings.DOMAIN_WIDTH; i+=r0){
        boundaryParticles.push_back(BoundaryParticle(i, 0.f, 0.f, 1.f));
        boundaryParticles.push_back(BoundaryParticle(i, settings.DOMAIN_HEIGHT, 0.f, -1.f));
    }
    

    particlesBufferA.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    particlesBufferB.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    settingsBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    boundaryParticlesBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        particlesBufferA[i] = m_core->bufferFromData(particles.data(),sizeof(Particle) * particles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        particlesBufferB[i] = m_core->bufferFromData(particles.data(),sizeof(Particle) * particles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        settingsBuffer[i]   = m_core->bufferFromData(&settings, sizeof(SPHSettings), vk::BufferUsageFlagBits::eUniformBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU);
        boundaryParticlesBuffer[i] = m_core->bufferFromData(boundaryParticles.data(),sizeof(Particle) * boundaryParticles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
    }
    initFrameResources();
    createDescriptorPool();
    createDescriptorSetLayout();
    createDescriptorSets();
    
    boundaryUpdatePass = gpu::ComputePass(m_core, SHADER_PATH"/boundary.comp", descriptorSetLayout);
    initPass = gpu::ComputePass(m_core, SHADER_PATH"/init.comp", descriptorSetLayout);
    predictDensityPass = gpu::ComputePass(m_core, SHADER_PATH"/predict_density.comp", descriptorSetLayout);
    predictStressPass = gpu::ComputePass(m_core, SHADER_PATH"/predict_stress.comp", descriptorSetLayout);
    predictForcePass = gpu::ComputePass(m_core, SHADER_PATH"/predict_force.comp", descriptorSetLayout);
    applyPass = gpu::ComputePass(m_core, SHADER_PATH"/apply.comp", descriptorSetLayout);
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

    boundaryUpdatePass.destroy();
    initPass.destroy();
    predictStressPass.destroy();
    predictDensityPass.destroy();
    predictForcePass.destroy();
    applyPass.destroy();

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


void GranularMatter::updateSettings(float dt, int currentFrame){

    settings.dt = dt; 
                        
    void* mappedData = m_core->mapBuffer(settingsBuffer[currentFrame]);
    memcpy(mappedData, &settings, (size_t) sizeof(SPHSettings));
    m_core->flushBuffer(settingsBuffer[currentFrame], 0, (size_t) sizeof(SPHSettings));
    m_core->unmapBuffer(settingsBuffer[currentFrame]);

}

void GranularMatter::update(int currentFrame, int imageIndex){
    auto currentTime = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    startTime = std::chrono::high_resolution_clock::now();
    std::cout << dt << std::endl;
    // float accumulator = dt;
    // float stepSize = 1.f/120.f;
    
    updateSettings(dt, currentFrame);

    // while( accumulator >= dt ){
    vk::CommandBufferBeginInfo beginInfo;
    try{
        commandBuffers[currentFrame].begin(beginInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }

    //* Copy data from last frame B to current frame A 
    {
        // Todo: try with mutiple descriptor sets
        vk::BufferCopy copyRegion(0, 0, sizeof(Particle) * particles.size());
        commandBuffers[currentFrame].copyBuffer(particlesBufferB[(currentFrame - 1) % gpu::MAX_FRAMES_IN_FLIGHT], particlesBufferA[currentFrame], 1, &copyRegion);
    }
    //* Wait for copy action
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);
    
    //* compute boundary densities
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, boundaryUpdatePass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, boundaryUpdatePass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].dispatch((uint32_t)boundaryParticles.size(), 1, 1);
    }
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);

    if(simulationRunning){
        //* init pass
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, initPass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, initPass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
        }
        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);
        
        //? maybe add dynamic loop here
        //* Do 3 iterations
        for (size_t i = 0; i < 1; i++)
        {
            //* predict density
            {
                commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, predictDensityPass.m_pipeline);
                commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictDensityPass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
                commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
            }
            //* wait for compute pass
            commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);
            //* predict stress
            {
                commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, predictStressPass.m_pipeline);
                commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictStressPass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
                commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
            }
            //* wait for compute pass
            commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);
            //* predict force
            {
                commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, predictForcePass.m_pipeline);
                commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictForcePass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
                commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
            }
            //* wait for compute pass
            commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, nullptr);      
            //* Copy data from A to B 
            {
                vk::BufferCopy copyRegion(0, 0, sizeof(Particle) * particles.size());
                commandBuffers[currentFrame].copyBuffer(particlesBufferA[currentFrame], particlesBufferB[currentFrame], 1, &copyRegion);
            }
            //* Wait for copy action
            commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);
        }
        //* apply force
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, applyPass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, applyPass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
        }
        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);   

    }
    else{
        {
            // Todo: try with mutiple descriptor sets
            vk::BufferCopy copyRegion(0, 0, sizeof(Particle) * particles.size());
            commandBuffers[currentFrame].copyBuffer(particlesBufferA[currentFrame], particlesBufferB[currentFrame], 1, &copyRegion);
        }
        //* Wait for copy action
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);
        
    }

    //* submit calls
    try{
        commandBuffers[currentFrame].end();
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }
    //     accumulator -= stepSize;
    // }
}

void GranularMatter::createDescriptorSetLayout() {
    vk::DescriptorSetLayoutBinding particlesLayoutBindingIn(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
    vk::DescriptorSetLayoutBinding particlesLayoutBindingOut(1, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
    vk::DescriptorSetLayoutBinding settingsLayoutBinding(2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
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
