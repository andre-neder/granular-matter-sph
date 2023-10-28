#include "granular_matter.h"
#include <chrono>
#include "iostream"
#include "global.h"

BitonicSortParameters params;
uint32_t workgroup_size_x;
uint32_t n;
uint32_t workGroupCountSort;
uint32_t workGroupCount;

glm::ivec3 computeSpace = glm::ivec3(64, 32, 1);

#define TIMESTAMP_QUERY_COUNT 9

uint32_t nextQueryIndex = 0;
uint32_t nextQuery(){
    nextQueryIndex++;
    return nextQueryIndex - 1;
}

std::vector<vk::QueryPool> timeQueryPools;
std::vector<std::string> passLabels = {
    // "Boundary               ",
    "Copy last frame data   ",
    "Init                   ",
    // "Grid quantification    ",
    "Sorting                ",
    "Start indices          ",
    "Density                ",
    "Stress                 ",
    "Force                  ",
    "Integration            "
};

GranularMatter::GranularMatter(gpu::Core* core)
{
    m_core = core;

    timeQueryPools.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        vk::QueryPoolCreateInfo createInfo{
            {},
            vk::QueryType::eTimestamp,
            TIMESTAMP_QUERY_COUNT, // start time & end time
            {}
        };
        
        vk::Result result = m_core->getDevice().createQueryPool(&createInfo, nullptr, &timeQueryPools[i]);
        if (result != vk::Result::eSuccess)
        {
            throw std::runtime_error("Failed to create time query pool!");
        }
    }
   
    // equilibrium distance
    float r0 = 0.5f * settings.kernelRadius;

    float initialDistance = 0.5f * settings.kernelRadius;

    for(int i = 0;i < computeSpace.x ; i++){
        for(int j = 0;j < computeSpace.y ; j++){
            particles.push_back(Particle(i * initialDistance + settings.kernelRadius  + (settings.DOMAIN_WIDTH / 2 - initialDistance * computeSpace.x / 2) ,j * initialDistance + settings.kernelRadius ));
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

    particlesBufferA.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    particlesBufferB.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    // settingsBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    boundaryParticlesBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    particleCellBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    startingIndicesBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        particlesBufferA[i] = m_core->bufferFromData(particles.data(),sizeof(Particle) * particles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        particlesBufferB[i] = m_core->bufferFromData(particles.data(),sizeof(Particle) * particles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        // settingsBuffer[i]   = m_core->bufferFromData(&settings, sizeof(SPHSettings), vk::BufferUsageFlagBits::eUniformBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_CPU_TO_GPU);
        boundaryParticlesBuffer[i] = m_core->bufferFromData(boundaryParticles.data(),sizeof(Particle) * boundaryParticles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        particleCellBuffer[i] = m_core->bufferFromData(particleCells.data(), sizeof(ParticleGridEntry) * particleCells.size(),vk::BufferUsageFlagBits::eStorageBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
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
    vk::PhysicalDeviceLimits limits = m_core->getPhysicalDevice().getProperties().limits;

    n = (uint32_t)particleCells.size();
    std::cout << "Particle count: " << n << std::endl;
    workgroup_size_x = 1;
    auto maxWorkGroupInvocations = (uint32_t)32;
    if(n < maxWorkGroupInvocations * 2){
        workgroup_size_x = n / 2;
    }
    else{
        workgroup_size_x = maxWorkGroupInvocations;
    }

    workGroupCountSort = n / ( workgroup_size_x * 2 );
    workGroupCount = n / workgroup_size_x;

    initPass = gpu::ComputePass(m_core, SHADER_PATH"/init.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workgroup_size_x) }, sizeof(SPHSettings));
    bitonicSortPass = gpu::ComputePass(m_core, SHADER_PATH"/bitonic_sort.comp", descriptorSetLayoutsCell, { gpu::SpecializationConstant(1, workgroup_size_x) }, sizeof(BitonicSortParameters));
    startingIndicesPass = gpu::ComputePass(m_core, SHADER_PATH"/start_indices.comp", descriptorSetLayoutsCell, { gpu::SpecializationConstant(1, workgroup_size_x) }); // , { gpu::SpecializationConstant(1, workgroup_size_x) }

    predictDensityPass = gpu::ComputePass(m_core, SHADER_PATH"/predict_density.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workgroup_size_x) }, sizeof(SPHSettings));
    predictStressPass = gpu::ComputePass(m_core, SHADER_PATH"/predict_stress.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workgroup_size_x) }, sizeof(SPHSettings));
    predictForcePass = gpu::ComputePass(m_core, SHADER_PATH"/predict_force.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workgroup_size_x) }, sizeof(SPHSettings));
    applyPass = gpu::ComputePass(m_core, SHADER_PATH"/apply.comp", descriptorSetLayoutsParticle, { gpu::SpecializationConstant(1, workgroup_size_x) }, sizeof(SPHSettings));

    boundaryUpdatePass = gpu::ComputePass(m_core, SHADER_PATH"/boundary.comp", descriptorSetLayoutsParticle, sizeof(SPHSettings));

    auto boundaryUpdateCommandBuffer = m_core->beginSingleTimeCommands();
     //* compute boundary densities
    {
        boundaryUpdateCommandBuffer.bindPipeline(vk::PipelineBindPoint::eCompute, boundaryUpdatePass.m_pipeline);
        boundaryUpdateCommandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eCompute, boundaryUpdatePass.m_pipelineLayout, 0, 1, &descriptorSets[0], 0, nullptr);
        boundaryUpdateCommandBuffer.pushConstants(boundaryUpdatePass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
        boundaryUpdateCommandBuffer.dispatch((uint32_t)boundaryParticles.size(), 1, 1);
        boundaryUpdateCommandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTopOfPipe, {}, {}, nullptr, nullptr);

    }
    core->endSingleTimeCommands(boundaryUpdateCommandBuffer);
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

    bitonicSortPass.destroy();
    startingIndicesPass.destroy();
    boundaryUpdatePass.destroy();
    initPass.destroy();
    predictStressPass.destroy();
    predictDensityPass.destroy();
    predictForcePass.destroy();
    applyPass.destroy();
    
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        m_core->getDevice().destroyQueryPool(timeQueryPools[i]);
    }

    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        m_core->destroyBuffer(particlesBufferA[i]);
        m_core->destroyBuffer(particlesBufferB[i]);
        // m_core->destroyBuffer(settingsBuffer[i]);
        m_core->destroyBuffer(boundaryParticlesBuffer[i]);
        m_core->destroyBuffer(particleCellBuffer[i]);
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


void GranularMatter::update(int currentFrame, int imageIndex){ 

    uint64_t buffer[TIMESTAMP_QUERY_COUNT];

    vk::Result result = m_core->getDevice().getQueryPoolResults(timeQueryPools[currentFrame], 0, TIMESTAMP_QUERY_COUNT, sizeof(uint64_t) * TIMESTAMP_QUERY_COUNT, buffer, sizeof(uint64_t), vk::QueryResultFlagBits::e64);
    if (result == vk::Result::eNotReady)
    {
        
    }
    else if (result == vk::Result::eSuccess)
    {
        //  std::cout << "################# Timimgs ##################" << std::endl;
        for (size_t i = 0; i < TIMESTAMP_QUERY_COUNT -1; i++) {
            // std::cout << passLabels[i] << (buffer[i + 1] - buffer[i]) / (float)1000000 << " ms" << std::endl;
            passTimeings.push_back(passLabels[i] + std::to_string((buffer[i + 1] - buffer[i]) / (float)1000000) + " ms");
        }
        passTimeings.push_back("Total                  " + std::to_string((buffer[TIMESTAMP_QUERY_COUNT -1] - buffer[0]) / (float)1000000) + " ms");
    }
    else
    {
        throw std::runtime_error("Failed to receive query results!");
    }
        
    auto currentTime = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
    startTime = std::chrono::high_resolution_clock::now();
    // std::cout << dt << std::endl;
    // float accumulator = dt;
    // float stepSize = 1.f/120.f;
    
 
    // settings.dt = dt;

    // Courant-Friedrichs–Lewy (CFL) condition
    float area = settings.particleRadius * 2.f;
    float v_max = sqrt((2 * settings.mass * settings.g.length()) / (settings.rhoAir * area * settings.dragCoefficient));
    float C_courant = 0.5f;
    float dt_max = C_courant * (settings.kernelRadius / v_max);
    // std::cout << dt << " " << 1.f / 60.f << " " <<  1.f/120.f << " " << dt_max << std::endl;
    settings.dt = std::min(dt, dt_max);

    vk::MemoryBarrier writeReadBarrier{
        vk::AccessFlagBits::eMemoryWrite,
        vk::AccessFlagBits::eMemoryRead
    };

    // while( accumulator >= dt ){
    vk::CommandBufferBeginInfo beginInfo;
    try{
        commandBuffers[currentFrame].begin(beginInfo);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }

     // Queries must be reset after each individual use.
    commandBuffers[currentFrame].resetQueryPool(timeQueryPools[currentFrame], 0, TIMESTAMP_QUERY_COUNT);
    nextQueryIndex = 0;
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, timeQueryPools[currentFrame], nextQuery());
    //Todo: Do this only once before loop 
    //* compute boundary densities
    // {
    //     commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, boundaryUpdatePass.m_pipeline);
    //     commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, boundaryUpdatePass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
    //     commandBuffers[currentFrame].dispatch((uint32_t)boundaryParticles.size(), 1, 1);
    // }
    
    // //* wait for compute pass
    // commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
    // commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    //* Copy data from last frame B to current frame A 
    {
        // Todo: try with mutiple descriptor sets
        vk::BufferCopy copyRegion(0, 0, sizeof(Particle) * particles.size());
        commandBuffers[currentFrame].copyBuffer(particlesBufferB[(currentFrame - 1) % gpu::MAX_FRAMES_IN_FLIGHT], particlesBufferA[currentFrame], 1, &copyRegion);
    }
    
    //* Wait for copy action
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    //* init pass A -> B
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, initPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, initPass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, initPass.m_pipelineLayout, 1, 1, &descriptorSetsCell[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].pushConstants(initPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
        commandBuffers[currentFrame].dispatch(workGroupCount, 1, 1);
    }
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
 
    //* Sort cell hashes
    {   
        //? https://poniesandlight.co.uk/reflect/bitonic_merge_sort/
        //? https://github.com/tgfrerer/island/blob/wip/apps/examples/bitonic_merge_sort_example/bitonic_merge_sort_example_app/bitonic_merge_sort_example_app.cpp
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, bitonicSortPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, bitonicSortPass.m_pipelineLayout, 0, 1, &descriptorSetsCell[currentFrame], 0, nullptr);

        auto dispatch = [ & ]( uint32_t h ) {
		    params.h = h;

            commandBuffers[currentFrame].pushConstants(bitonicSortPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(BitonicSortParameters), &params);
            commandBuffers[currentFrame].dispatch( workGroupCountSort, 1, 1 );
            commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
            
        };

        auto local_bms = [ & ]( uint32_t h ) {
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
		assert( (h != 0) && ((h & (h - 1)) == 0) );

		local_bms( h );
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
    
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    //* get starting indices
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, startingIndicesPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, startingIndicesPass.m_pipelineLayout, 0, 1, &descriptorSetsCell[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].dispatch(workGroupCount, 1, 1);
    }

    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    //* predict density B -> A
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, predictDensityPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictDensityPass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictDensityPass.m_pipelineLayout, 1, 1, &descriptorSetsCell[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].pushConstants(predictDensityPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
        commandBuffers[currentFrame].dispatch(workGroupCount, 1, 1);
    }
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    //* predict stress A -> B
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, predictStressPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictStressPass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictStressPass.m_pipelineLayout, 1, 1, &descriptorSetsCell[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].pushConstants(predictStressPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
        commandBuffers[currentFrame].dispatch(workGroupCount, 1, 1);
    }
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    //* predict force B -> A
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, predictForcePass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictForcePass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictForcePass.m_pipelineLayout, 1, 1, &descriptorSetsCell[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].pushConstants(predictForcePass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
        commandBuffers[currentFrame].dispatch(workGroupCount, 1, 1);
    }
    
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    //* When simulation is running or should proceed one step apply calculated forces
    if(simulationRunning || simulationStepForward){
        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);      
        //* apply force A -> B
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, applyPass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, applyPass.m_pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].pushConstants(applyPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
            commandBuffers[currentFrame].dispatch(workGroupCount, 1, 1);
        }
        simulationStepForward = false;

        //* Wait for copy action
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eVertexInput, {}, writeReadBarrier, nullptr, nullptr);
        commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    }
    else{
        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer , {}, writeReadBarrier, nullptr, nullptr);
        //* copy A -> B
        {
            // Todo: try with mutiple descriptor sets
            vk::BufferCopy copyRegion(0, 0, sizeof(Particle) * particles.size());
            commandBuffers[currentFrame].copyBuffer(particlesBufferA[currentFrame], particlesBufferB[currentFrame], 1, &copyRegion);
        }
        //* Wait for copy action
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eVertexInput, {}, writeReadBarrier, nullptr, nullptr);
        commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    }

    assert(nextQuery() == TIMESTAMP_QUERY_COUNT);
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
        // vk::DescriptorSetLayoutBinding buffer2(1, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
        vk::DescriptorSetLayoutBinding buffer3(2, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings = {
            buffer,
            // buffer2,
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
        // vk::DescriptorSetLayoutBinding settingsLayoutBinding(2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);
        vk::DescriptorSetLayoutBinding boundaryLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);

        std::array<vk::DescriptorSetLayoutBinding, 3> bindings = {
            particlesLayoutBindingIn, 
            particlesLayoutBindingOut,
            // settingsLayoutBinding,
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

            vk::DescriptorBufferInfo bufferInfo3(startingIndicesBuffers[i], 0, sizeof(uint32_t) * startingIndices.size());
            vk::WriteDescriptorSet descriptorWrite3(descriptorSetsCell[i], 2, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &bufferInfo3);

            std::array<vk::WriteDescriptorSet, 2> descriptorWrites{
                descriptorWrite1,
                // descriptorWrite2,
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
            // vk::DescriptorBufferInfo bufferInfoSettings(settingsBuffer[i], 0, sizeof(SPHSettings));
            vk::DescriptorBufferInfo bufferInfoBoundary(boundaryParticlesBuffer[i], 0, sizeof(Particle) * boundaryParticles.size());
        
            vk::WriteDescriptorSet descriptorWriteA(descriptorSets[i], 0, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &bufferInfoA);
            vk::WriteDescriptorSet descriptorWriteB(descriptorSets[i], 1, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &bufferInfoB);
            // vk::WriteDescriptorSet descriptorWriteSettings(descriptorSets[i], 2, 0, 1, vk::DescriptorType::eUniformBuffer, {}, &bufferInfoSettings);
            vk::WriteDescriptorSet descriptorWriteBoundary(descriptorSets[i], 3, 0, 1, vk::DescriptorType::eStorageBuffer, {}, &bufferInfoBoundary);

            std::array<vk::WriteDescriptorSet, 3> descriptorWrites{
                descriptorWriteA, 
                descriptorWriteB,
                // descriptorWriteSettings,
                descriptorWriteBoundary
            };
            
            m_core->getDevice().updateDescriptorSets(descriptorWrites, nullptr);
        }
    }
}

void GranularMatter::createDescriptorPool() {
    {
        vk::DescriptorPoolSize bufferSize(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        // vk::DescriptorPoolSize bufferSize2(vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        vk::DescriptorPoolSize bufferSize3(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        std::array<vk::DescriptorPoolSize, 2> poolSizes{
            bufferSize,
            // bufferSize2,
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
        // vk::DescriptorPoolSize settingsSize(vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        vk::DescriptorPoolSize boundarySize(vk::DescriptorType::eStorageBuffer, static_cast<uint32_t>(gpu::MAX_FRAMES_IN_FLIGHT));
        std::array<vk::DescriptorPoolSize, 3> poolSizes{
            particlesInSize, 
            particlesOutSize,
            // settingsSize,
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
