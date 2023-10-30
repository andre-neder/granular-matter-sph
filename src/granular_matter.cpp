#include "granular_matter.h"
#include <chrono>
#include "iostream"
#include "global.h"

#include <Discregrid/All>

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

double bottomBorderFn(Eigen::Vector3d const& wtf){
    return 0.0;
}

GranularMatter::GranularMatter(gpu::Core* core)
{
    m_core = core;

    
    // Eigen::AlignedBox3d domain;
    // domain.extend(Eigen::AlignedBox3d(Eigen::Vector3d(0,0,0), Eigen::Vector3d(settings.DOMAIN_WIDTH,settings.DOMAIN_HEIGHT,1)));
    // std::array<unsigned int, 3> resolution = {{10, 10, 10}};
    // Discregrid::CubicLagrangeDiscreteGrid sdf(domain, resolution);

    // Discregrid::DiscreteGrid::ContinuousFunction func1 = bottomBorderFn;
    // auto df_index1 = sdf.addFunction(func1);

    // auto val1 = sdf.interpolate(df_index1, {0.1, 0.2, 0.0});
    // Eigen::Vector3d grad2;
    // auto val2 = sdf->interpolate(df_index2, {0.3, 0.2, 0.1}, &grad2);

    // sdf.reduce_field(df_index1, [](Eigen::Vector3d const& x, double v)
    // {
    // 	return x.x() < 0.0 && v > 0.0;
    // });

    // sdf.save("filename.sdf");
    // sdf.load(filename.sdf); 
    // sdf = Discregrid::CubicLagrangeDiscreteGrid(filename.sdf);


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
    
    particleCells.resize(particles.size());
    std::fill(particleCells.begin(), particleCells.end(), ParticleGridEntry());
    startingIndices.resize(particles.size());
    std::fill(startingIndices.begin(), startingIndices.end(), UINT32_MAX);

    particlesBufferA.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    particlesBufferB.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    particleCellBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    startingIndicesBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        particlesBufferA[i] = m_core->bufferFromData(particles.data(),sizeof(Particle) * particles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        particlesBufferB[i] = m_core->bufferFromData(particles.data(),sizeof(Particle) * particles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        particleCellBuffer[i] = m_core->bufferFromData(particleCells.data(), sizeof(ParticleGridEntry) * particleCells.size(),vk::BufferUsageFlagBits::eStorageBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
        startingIndicesBuffers [i] = m_core->bufferFromData(startingIndices.data(), sizeof(uint32_t) * startingIndices.size(),vk::BufferUsageFlagBits::eStorageBuffer, VmaMemoryUsage::VMA_MEMORY_USAGE_GPU_ONLY);
    }
    initFrameResources();
    createDescriptorPool();
    createDescriptorSetLayout();
    createDescriptorSets();
    
    std::vector<vk::DescriptorSetLayout> descriptorSetLayoutsParticle{
        descriptorSetLayoutParticles
    };
    std::vector<vk::DescriptorSetLayout> descriptorSetLayoutsParticleCell{
        descriptorSetLayoutParticles,
        descriptorSetLayoutGrid
    };
    std::vector<vk::DescriptorSetLayout> descriptorSetLayoutsCell{
        descriptorSetLayoutGrid
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
        m_core->destroyBuffer(particleCellBuffer[i]);
        m_core->destroyBuffer(startingIndicesBuffers[i]);
    }

    m_core->destroyDescriptorSetLayout(descriptorSetLayoutGrid);
    m_core->destroyDescriptorSetLayout(descriptorSetLayoutParticles);
    
    m_core->destroyDescriptorPool(descriptorPool);
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
        for (size_t i = 0; i < TIMESTAMP_QUERY_COUNT -1; i++) {
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

    //* Copy data from last frame B to current frame A 
    {
        vk::BufferCopy copyRegion(0, 0, sizeof(Particle) * particles.size());
        commandBuffers[currentFrame].copyBuffer(particlesBufferB[(currentFrame - 1) % gpu::MAX_FRAMES_IN_FLIGHT], particlesBufferA[currentFrame], 1, &copyRegion);
    }
    
    //* Wait for copy action
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    //* init pass A -> B
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, initPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, initPass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, initPass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
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
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, bitonicSortPass.m_pipelineLayout, 0, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);

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
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, startingIndicesPass.m_pipelineLayout, 0, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].dispatch(workGroupCount, 1, 1);
    }

    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    //* predict density B -> A
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, predictDensityPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictDensityPass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictDensityPass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].pushConstants(predictDensityPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
        commandBuffers[currentFrame].dispatch(workGroupCount, 1, 1);
    }
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    //* predict stress A -> B
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, predictStressPass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictStressPass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictStressPass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].pushConstants(predictStressPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
        commandBuffers[currentFrame].dispatch(workGroupCount, 1, 1);
    }
    //* wait for compute pass
    commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], nextQuery());
    
    //* predict force B -> A
    {
        commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, predictForcePass.m_pipeline);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictForcePass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, predictForcePass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
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
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, applyPass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
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
    
    descriptorSetLayoutGrid = m_core->createDescriptorSetLayout({
        {0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute}
    });

    descriptorSetLayoutParticles = m_core->createDescriptorSetLayout({
        {0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute},
        {1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute}
    });
 
}

void GranularMatter::createDescriptorSets() {

    descriptorSetsGrid = m_core->allocateDescriptorSets(descriptorSetLayoutGrid, descriptorPool, gpu::MAX_FRAMES_IN_FLIGHT);
    
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        m_core->updateDescriptorSet(descriptorSetsGrid[i], {
            { 0, vk::DescriptorType::eStorageBuffer, particleCellBuffer[i], sizeof(ParticleGridEntry) * particleCells.size() },
            { 2, vk::DescriptorType::eStorageBuffer, startingIndicesBuffers[i], sizeof(uint32_t) * startingIndices.size() }
        });
    }



    descriptorSetsParticles = m_core->allocateDescriptorSets(descriptorSetLayoutParticles, descriptorPool, gpu::MAX_FRAMES_IN_FLIGHT);
    
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        m_core->updateDescriptorSet(descriptorSetsParticles[i], {
            { 0, vk::DescriptorType::eStorageBuffer, particlesBufferA[i], sizeof(Particle) * particles.size() },
            { 1, vk::DescriptorType::eStorageBuffer, particlesBufferB[i], sizeof(Particle) * particles.size() }
        });
    }

}

void GranularMatter::createDescriptorPool() {

    descriptorPool = m_core->createDescriptorPool({
        { vk::DescriptorType::eStorageBuffer, 5 * gpu::MAX_FRAMES_IN_FLIGHT }
    }, 2 * gpu::MAX_FRAMES_IN_FLIGHT );
}
