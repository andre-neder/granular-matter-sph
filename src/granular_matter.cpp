#include "granular_matter.h"
#include <chrono>
#include "iostream"
#include "global.h"

BitonicSortParameters params;

GranularMatter::GranularMatter(gpu::Core* core)
{
    m_core = core;
    
    // particles = std::vector<Particle>(computeSpace.x * computeSpace.y * computeSpace.z);
    // std::generate(particles.begin(), particles.end(), [this]() {
    //     return Particle(((static_cast<float>(std::rand()) / RAND_MAX) / 4.f + 0.375f) * settings.DOMAIN_WIDTH, ((static_cast<float>(std::rand()) / RAND_MAX  / 2.f )) * settings.DOMAIN_HEIGHT);
    // });
    // equilibrium distance
    float r0 = 0.5f * settings.kernelRadius;

    float initialDistance = 0.9f * settings.kernelRadius;

    for(int i = 0;i < computeSpace.x ; i++){
        for(int j = 0;j < computeSpace.y ; j++){
            particles.push_back(Particle(i * initialDistance + r0  ,j * initialDistance + r0 ));
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
    
    particleCells.resize(particles.size());
    std::fill(particleCells.begin(), particleCells.end(), ParticleGridEntry());
    startingIndices.resize(particles.size());
    std::fill(startingIndices.begin(), startingIndices.end(), UINT32_MAX);

    bitonicSortParameterBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    particlesBufferA.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    particlesBufferB.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    settingsBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    boundaryParticlesBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    particleCellBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    startingIndicesBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        particlesBufferA[i] = m_core->bufferFromData(particles.data(),sizeof(Particle) * particles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        particlesBufferB[i] = m_core->bufferFromData(particles.data(),sizeof(Particle) * particles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        settingsBuffer[i]   = m_core->bufferFromData(&settings, sizeof(SPHSettings), vk::BufferUsageFlagBits::eUniformBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU);
        boundaryParticlesBuffer[i] = m_core->bufferFromData(boundaryParticles.data(),sizeof(Particle) * boundaryParticles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        particleCellBuffer[i] = m_core->bufferFromData(particleCells.data(), sizeof(ParticleGridEntry) * particleCells.size(),vk::BufferUsageFlagBits::eStorageBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        bitonicSortParameterBuffers[i] = m_core->bufferFromData(&params, sizeof(BitonicSortParameters), vk::BufferUsageFlagBits::eUniformBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU);
        startingIndicesBuffers [i] = m_core->bufferFromData(startingIndices.data(), sizeof(uint32_t) * startingIndices.size(),vk::BufferUsageFlagBits::eStorageBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
    }
    initFrameResources();
    createDescriptorPool();
    createDescriptorSetLayout();
    createDescriptorSets();
    
    std::vector<vk::DescriptorSetLayout> descriptorSetLayoutsParticle{
        descriptorSetLayout
    };
    std::vector<vk::DescriptorSetLayout> descriptorSetLayoutsParticleCell{
        descriptorSetLayout,
        descriptorSetLayoutCell
    };
    std::vector<vk::DescriptorSetLayout> descriptorSetLayoutsCell{
        descriptorSetLayoutCell
    };

    //* Sorting stuff
    const vk::PhysicalDeviceLimits limits = m_core->getPhysicalDevice().getProperties().limits;
    const uint32_t n = (uint32_t)particles.size();
    uint32_t workgroup_size_x = 1;
    if(n < limits.maxComputeWorkGroupInvocations * 2){
        workgroup_size_x = n / 2;
    }
    else{
        workgroup_size_x = limits.maxComputeWorkGroupInvocations;
    }


    neighborhoodUpdatePass = gpu::ComputePass(m_core, SHADER_PATH"/neighborhood_update.comp", descriptorSetLayoutsParticleCell);
    bitonicSortPass = gpu::ComputePass(m_core, SHADER_PATH"/bitonic_sort.comp", descriptorSetLayoutsCell, { gpu::SpecializationConstant(1, workgroup_size_x) });
    startingIndicesPass = gpu::ComputePass(m_core, SHADER_PATH"/start_indices.comp", descriptorSetLayoutsCell); // , { gpu::SpecializationConstant(1, workgroup_size_x) }

    boundaryUpdatePass = gpu::ComputePass(m_core, SHADER_PATH"/boundary.comp", descriptorSetLayoutsParticle);

    initPass = gpu::ComputePass(m_core, SHADER_PATH"/init.comp", descriptorSetLayoutsParticle);
    predictDensityPass = gpu::ComputePass(m_core, SHADER_PATH"/predict_density.comp", descriptorSetLayoutsParticleCell);
    predictStressPass = gpu::ComputePass(m_core, SHADER_PATH"/predict_stress.comp", descriptorSetLayoutsParticle);
    predictForcePass = gpu::ComputePass(m_core, SHADER_PATH"/predict_force.comp", descriptorSetLayoutsParticle);
    applyPass = gpu::ComputePass(m_core, SHADER_PATH"/apply.comp", descriptorSetLayoutsParticle);
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

    neighborhoodUpdatePass.destroy();
    bitonicSortPass.destroy();
    startingIndicesPass.destroy();
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
        m_core->destroyBuffer(particleCellBuffer[i]);
        m_core->destroyBuffer(bitonicSortParameterBuffers[i]);
        m_core->destroyBuffer(startingIndicesBuffers[i]);
    }
    device.destroyDescriptorSetLayout(descriptorSetLayoutCell);
    device.destroyDescriptorPool(descriptorPoolCell);
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
    
    //* compute cell hashes
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, neighborhoodUpdatePass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, neighborhoodUpdatePass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, neighborhoodUpdatePass.m_pipelineLayout, 1, 1, &descriptorSetsCell[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].dispatch((uint32_t)particleCells.size(), 1, 1);
    }
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);

    //* Sort cell hashes
    {   
        //? https://poniesandlight.co.uk/reflect/bitonic_merge_sort/
        //? https://github.com/tgfrerer/island/blob/wip/apps/examples/bitonic_merge_sort_example/bitonic_merge_sort_example_app/bitonic_merge_sort_example_app.cpp
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, bitonicSortPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, bitonicSortPass.m_pipelineLayout, 0, 1, &descriptorSetsCell[currentFrame], 0, nullptr);
            
        const vk::PhysicalDeviceLimits limits = m_core->getPhysicalDevice().getProperties().limits;
        const uint32_t n = (uint32_t)particles.size();
        uint32_t workgroup_size_x = 1;
        if(n < limits.maxComputeWorkGroupInvocations * 2){
            workgroup_size_x = n / 2;
        }
        else{
            workgroup_size_x = limits.maxComputeWorkGroupInvocations;
        }
        const uint32_t workgroup_count = n / ( workgroup_size_x * 2 );

        auto dispatch = [ & ]( uint32_t h ) {
		    params.h = h;

            void* mappedData = m_core->mapBuffer(bitonicSortParameterBuffers[currentFrame]);
            memcpy(mappedData, &params, (size_t) sizeof(BitonicSortParameters));
            m_core->flushBuffer(bitonicSortParameterBuffers[currentFrame], 0, (size_t) sizeof(BitonicSortParameters));
            m_core->unmapBuffer(bitonicSortParameterBuffers[currentFrame]);

            commandBuffers[currentFrame].dispatch( workgroup_count, 1, 1 );
            commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);

        };

        auto local_bitonic_merge_sort_example = [ & ]( uint32_t h ) {
            params.algorithm = BitonicSortParameters::eAlgorithmVariant::eLocalBitonicMergeSortExample;
            dispatch( h );
        };

        auto big_flip = [ & ]( uint32_t h ) {
            params.algorithm = BitonicSortParameters::eAlgorithmVariant::eBigFlip;
            dispatch( h );
        };

        auto local_disperse = [ & ]( uint32_t h ) {
            params.algorithm = BitonicSortParameters::eAlgorithmVariant::eLocalDisperse;
            dispatch( h );
        };

        auto big_disperse = [ & ]( uint32_t h ) {
            params.algorithm = BitonicSortParameters::eAlgorithmVariant::eBigDisperse;
            dispatch( h );
        };


        uint32_t h = workgroup_size_x * 2;
		assert( h <= n );
		assert( h % 2 == 0 );

		local_bitonic_merge_sort_example( h );
		// we must now double h, as this happens before every flip
		h *= 2;

		for ( ; h <= n; h *= 2 ) {
			big_flip( h );

			for ( uint32_t hh = h / 2; hh > 1; hh /= 2 ) {

				if ( hh <= workgroup_size_x * 2 ) {
					// We can fit all elements for a disperse operation into continuous shader
					// workgroup local memory, which means we can complete the rest of the
					// cascade using a single shader invocation.
					local_disperse( hh );
					break;
				} else {
					big_disperse( hh );
				}
			}
		}

    }

    //* get starting indices
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, startingIndicesPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, startingIndicesPass.m_pipelineLayout, 0, 1, &descriptorSetsCell[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].dispatch((uint32_t)startingIndices.size(), 1, 1);
    }

    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);

    //Todo: Do this only once before loop 
    //* compute boundary densities
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, boundaryUpdatePass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, boundaryUpdatePass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].dispatch((uint32_t)boundaryParticles.size(), 1, 1);
    }

    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);


    //* init pass A -> B
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, initPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, initPass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
    }
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);

    //* predict density B -> A
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, predictDensityPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictDensityPass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictDensityPass.m_pipelineLayout, 1, 1, &descriptorSetsCell[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
    }
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);
    //* predict stress A -> B
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, predictStressPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictStressPass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        // commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictStressPass.m_pipelineLayout, 1, 1, &descriptorSetsCell[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
    }
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);
    //* predict force B -> A
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, predictForcePass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictForcePass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
    }
    
    //* When simulation is running or should proceed one step apply calculated forces
    if(simulationRunning || simulationStepForward){
        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);      

        //* apply force A -> B
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, applyPass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, applyPass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].dispatch(computeSpace.x, computeSpace.y, computeSpace.z);
        }
        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, nullptr, nullptr, nullptr);   

        simulationStepForward = false;
    }
    else{
        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer, {}, nullptr, nullptr, nullptr);

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
    {
        vk::DescriptorSetLayoutBinding buffer(0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
        vk::DescriptorSetLayoutBinding buffer2(1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
        vk::DescriptorSetLayoutBinding buffer3(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
        std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {
            buffer,
            buffer2,
            buffer3
        };
        vk::DescriptorSetLayoutCreateInfo layoutInfo({}, bindings);
        try{
            descriptorSetLayoutCell = m_core->getDevice().createDescriptorSetLayout(layoutInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    {
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
}
// Todo: connect all simulation frames
void GranularMatter::createDescriptorSets() {
    {
        std::vector<vk::DescriptorSetLayout> layouts(gpu::MAX_FRAMES_IN_FLIGHT, descriptorSetLayoutCell);
        vk::DescriptorSetAllocateInfo allocInfo(descriptorPoolCell, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT), layouts.data());
        descriptorSetsCell.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        try{
            descriptorSetsCell = m_core->getDevice().allocateDescriptorSets(allocInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
        
        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {

            vk::DescriptorBufferInfo bufferInfo1(particleCellBuffer[i], 0, sizeof(ParticleGridEntry) * particleCells.size());
            vk::WriteDescriptorSet descriptorWrite1(descriptorSetsCell[i], 0, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &bufferInfo1);

            vk::DescriptorBufferInfo bufferInfo2(bitonicSortParameterBuffers[i], 0, sizeof(BitonicSortParameters));
            vk::WriteDescriptorSet descriptorWrite2(descriptorSetsCell[i], 1, 0, 1, vk::DescriptorType::eUniformBuffer, {}, &bufferInfo2);

            vk::DescriptorBufferInfo bufferInfo3(startingIndicesBuffers[i], 0, sizeof(uint32_t) * startingIndices.size());
            vk::WriteDescriptorSet descriptorWrite3(descriptorSetsCell[i], 2, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &bufferInfo3);

            std::array<vk::WriteDescriptorSet, 3> descriptorWrites{
                descriptorWrite1,
                descriptorWrite2,
                descriptorWrite3
            };
            
            m_core->getDevice().updateDescriptorSets(descriptorWrites, nullptr);
        }
    }
    {
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
}

void GranularMatter::createDescriptorPool() {
    {
        vk::DescriptorPoolSize bufferSize(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        vk::DescriptorPoolSize bufferSize2(vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        vk::DescriptorPoolSize bufferSize3(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        std::array<vk::DescriptorPoolSize, 3> poolSizes{
            bufferSize,
            bufferSize2,
            bufferSize3
        };

        vk::DescriptorPoolCreateInfo poolInfo({}, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT), poolSizes);
        try{
            descriptorPoolCell = m_core->getDevice().createDescriptorPool(poolInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    {
        vk::DescriptorPoolSize particlesInSize(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        vk::DescriptorPoolSize particlesOutSize(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        vk::DescriptorPoolSize settingsSize(vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        vk::DescriptorPoolSize boundarySize(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        std::array<vk::DescriptorPoolSize, 4> poolSizes{
            particlesInSize, 
            particlesOutSize,
            settingsSize,
            boundarySize
        };

        vk::DescriptorPoolCreateInfo poolInfo({}, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT), poolSizes);
        try{
            descriptorPool = m_core->getDevice().createDescriptorPool(poolInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
}
