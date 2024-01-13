#include "granular_matter.h"
#include <chrono>
#include "iostream"
#include "global.h"
#include "utils.h"

SimulationMetrics simulationMetrics = SimulationMetrics();

BitonicSortParameters params;
uint32_t workGroupSize;
uint32_t n;
uint32_t workGroupCountSort;
uint32_t workGroupCountLR;
uint32_t workGroupCountHR;

glm::ivec3 computeSpace = glm::ivec3(16, 32, 16);

std::vector<vk::QueryPool> timeQueryPools;
std::vector<std::vector<std::string>> timestampLabels;
std::vector<std::vector<uint64_t>> timestamps;

float RandomFloat(float a, float b) {
    float random = ((float) rand()) / (float) RAND_MAX;
    float diff = b - a;
    float r = random * diff;
    return a + r;
}

GranularMatter::GranularMatter(gpu::Core* core)
{
    m_core = core;

    createSignedDistanceFields();

    timeQueryPools.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    timestampLabels.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    timestamps.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        m_core->createTimestampQueryPool(&timeQueryPools[i]);
        timestampLabels[i] = std::vector<std::string>();
    }


    float initialDistance = 0.5f * settings.h_LR;
    std::vector<glm::vec3> hrParticleOffsets = {
        {0, 0, 0},
        {settings.r_LR, 0, 0},
        {-settings.r_LR, 0, 0},
        {0, settings.r_LR, 0},
        {0, -settings.r_LR, 0},
        {0, 0, settings.r_LR},
        {0, 0, -settings.r_LR},
    };

    for(int i = 0;i < computeSpace.x ; i++){
        for(int j = 0;j < computeSpace.y ; j++){
            for(int k = 0;k < computeSpace.z ; k++){
                //  + (settings.DOMAIN_WIDTH / 2 - initialDistance * computeSpace.x / 2)
                // glm::vec3 lrPosition = glm::vec3(
                //     i * initialDistance + settings.h_LR + (settings.DOMAIN_WIDTH / 2 - initialDistance * computeSpace.x / 2),
                //     j * initialDistance + settings.r_LR, 
                //     k * initialDistance + settings.h_LR + (settings.DOMAIN_WIDTH / 2 - initialDistance * computeSpace.z / 2));

                glm::vec3 lrPosition = glm::vec3(
                    -(initialDistance * computeSpace.x / 2) + i * initialDistance + (initialDistance / 2.f),
                     j * initialDistance + (initialDistance / 2.f) + settings.r_LR + 8.f, //  (initialDistance * computeSpace.y / 2) + 
                    -(initialDistance * computeSpace.z / 2) + k * initialDistance + (initialDistance / 2.f)
                );

                lrParticles.push_back(LRParticle(
                    lrPosition.x,// + RandomFloat(-settings.r_LR, settings.r_LR), 
                    lrPosition.y,// + RandomFloat(-settings.r_LR, settings.r_LR), 
                    lrPosition.z// + RandomFloat(-settings.r_LR, settings.r_LR)
                ));

                for(uint32_t l = 0; l < settings.n_HR; l++){
                    glm::vec3 offset = hrParticleOffsets[l % hrParticleOffsets.size()];
                    hrParticles.push_back(HRParticle(
                        lrPosition.x + offset.x + RandomFloat(-settings.r_LR, settings.r_LR), 
                        lrPosition.y + offset.y + RandomFloat(-settings.r_LR, settings.r_LR),
                        lrPosition.z + offset.z + RandomFloat(-settings.r_LR, settings.r_LR)
                    ));
                }
                
            }
        }
    }
    
    particleCells.resize(lrParticles.size());
    std::fill(particleCells.begin(), particleCells.end(), ParticleGridEntry());
    startingIndices.resize(lrParticles.size());
    std::fill(startingIndices.begin(), startingIndices.end(), UINT32_MAX);

    additionalDataBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        additionalDataBuffer[i] = m_core->bufferFromData(&additionalData,  sizeof(AdditionalData), vk::BufferUsageFlagBits::eStorageBuffer, vma::MemoryUsage::eAutoPreferHost, vma::AllocationCreateFlagBits::eHostAccessSequentialWrite );
    }

    volumeMapTransformsBuffer = m_core->bufferFromData(volumeMapTransforms.data(), volumeMapTransforms.size() * sizeof(VolumeMapTransform),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer, vma::MemoryUsage::eAutoPreferDevice);

    particlesBufferB = m_core->bufferFromData(lrParticles.data(),sizeof(LRParticle) * lrParticles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, vma::MemoryUsage::eAutoPreferDevice);
    particlesBufferHR = m_core->bufferFromData(hrParticles.data(),sizeof(HRParticle) * hrParticles.size(),vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc, vma::MemoryUsage::eAutoPreferDevice);
    
    particleCellBuffer = m_core->bufferFromData(particleCells.data(), sizeof(ParticleGridEntry) * particleCells.size(),vk::BufferUsageFlagBits::eStorageBuffer, vma::MemoryUsage::eAutoPreferDevice);
    startingIndicesBuffers = m_core->bufferFromData(startingIndices.data(), sizeof(uint32_t) * startingIndices.size(),vk::BufferUsageFlagBits::eStorageBuffer, vma::MemoryUsage::eAutoPreferDevice);
    
    initFrameResources();
    createDescriptorPool();
    createDescriptorSetLayout();
    createDescriptorSets();

    iisphFences.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    iisphSemaphores.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);
    vk::SemaphoreCreateInfo semaphoreInfo;
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        try{
            iisphFences[i] = m_core->getDevice().createFence(fenceInfo);
            iisphSemaphores[i] = m_core->getDevice().createSemaphore(semaphoreInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }

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

    n = (uint32_t)particleCells.size();
    std::cout << "LRParticle count: " << n << std::endl;
    std::cout << "HRParticle count: " << n * settings.n_HR << std::endl;

    workGroupSize = 1;
    if(n < m_core->getIdealWorkGroupSize() * 2){
        workGroupSize = n / 2;
    }
    else{
        workGroupSize = m_core->getIdealWorkGroupSize();
    }

    workGroupCountSort = n / ( workGroupSize * 2 );
    workGroupCountLR = n / workGroupSize;
    workGroupCountHR = (uint32_t)hrParticles.size() / workGroupSize;
    initPass = gpu::ComputePass(m_core, SHADER_PATH"/init.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workGroupSize) }, sizeof(SPHSettings));
    bitonicSortPass = gpu::ComputePass(m_core, SHADER_PATH"/bitonic_sort.comp", descriptorSetLayoutsCell, { gpu::SpecializationConstant(1, workGroupSize) }, sizeof(BitonicSortParameters));
    startingIndicesPass = gpu::ComputePass(m_core, SHADER_PATH"/start_indices.comp", descriptorSetLayoutsCell, { gpu::SpecializationConstant(1, workGroupSize) }); // , { gpu::SpecializationConstant(1, workGroupSize) }

    computeDensityPass = gpu::ComputePass(m_core, SHADER_PATH"/compute_density.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workGroupSize) }, sizeof(SPHSettings));
    
    iisphvAdvPass = gpu::ComputePass(m_core, SHADER_PATH"/iisph_v_adv.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workGroupSize) }, sizeof(SPHSettings));
    iisphRhoAdvPass = gpu::ComputePass(m_core, SHADER_PATH"/iisph_rho_adv.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workGroupSize) }, sizeof(SPHSettings));
    iisphdijpjSolvePass = gpu::ComputePass(m_core, SHADER_PATH"/iisph_solve_dijpj.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workGroupSize) }, sizeof(SPHSettings));
    iisphPressureSolvePass = gpu::ComputePass(m_core, SHADER_PATH"/iisph_solve_pressure.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workGroupSize) }, sizeof(SPHSettings));
    iisphSolveEndPass = gpu::ComputePass(m_core, SHADER_PATH"/iisph_solve_end.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workGroupSize) }, sizeof(SPHSettings));


    computeStressPass = gpu::ComputePass(m_core, SHADER_PATH"/compute_stress.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workGroupSize) }, sizeof(SPHSettings));
    computeInternalForcePass = gpu::ComputePass(m_core, SHADER_PATH"/compute_internal_force.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workGroupSize) }, sizeof(SPHSettings));
    integratePass = gpu::ComputePass(m_core, SHADER_PATH"/integrate.comp", descriptorSetLayoutsParticle, { gpu::SpecializationConstant(1, workGroupSize) }, sizeof(SPHSettings));
    advectionPass = gpu::ComputePass(m_core, SHADER_PATH"/hr_advection.comp", descriptorSetLayoutsParticleCell, { gpu::SpecializationConstant(1, workGroupSize) }, sizeof(SPHSettings));
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

void GranularMatter::initFrameResources(){
    createCommandBuffers();
}

void GranularMatter::update(int currentFrame, int imageIndex, float dt){ 

    if(resetSimulation){
        simulationRunning = false;
        resetSimulation = false;
        simulationStepForward = false;
      
        m_core->updateBufferData(particlesBufferB, lrParticles.data(), sizeof(LRParticle) * lrParticles.size());

        m_core->updateBufferData(particlesBufferHR, hrParticles.data(), sizeof(HRParticle) * hrParticles.size());
        
    }

    //Read timesteps
    timestamps[currentFrame] = m_core->getTimestampQueryPoolResults(&timeQueryPools[currentFrame]);

    // Courant-Friedrichs–Lewy (CFL) condition
    float C_courant = 0.4f; 
    float dt_max = C_courant * (settings.h_LR / settings.v_max);
    settings.dt = std::min(dt, dt_max);

    vk::MemoryBarrier writeReadBarrier{
        vk::AccessFlagBits::eMemoryWrite,
        vk::AccessFlagBits::eMemoryRead
    };

    m_core->beginCommands(commandBuffers[currentFrame]);

    // Reset query labels and pool
    timestampLabels[currentFrame] = std::vector<std::string>(); 
    commandBuffers[currentFrame].resetQueryPool(timeQueryPools[currentFrame], 0, gpu::MAX_QUERY_POOL_COUNT);
    commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eTopOfPipe, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());

    //* When simulation is running or should proceed one step apply calculated forces
    if(simulationRunning || simulationStepForward){
        timestampLabels[currentFrame].push_back("Init");
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, initPass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, initPass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, initPass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].pushConstants(initPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
            commandBuffers[currentFrame].dispatch(workGroupCountLR, 1, 1);
        }
        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
        commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());
        
        timestampLabels[currentFrame].push_back("Neighborhood list sorting");
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

            uint32_t h = workGroupSize * 2;
            assert( h <= n );
            assert( h % 2 == 0 );
            assert( (h != 0) && ((h & (h - 1)) == 0) );

            local_bms( h );
            h *= 2;
            for ( ; h <= n; h *= 2 ) {
                big_flip( h );

                for ( uint32_t hh = h / 2; hh > 1; hh /= 2 ) {

                    if ( hh <= workGroupSize * 2 ) {
                        local_disperse( hh );
                        break;
                    } else {
                        big_disperse( hh );
                    }
                }
            }
        }
        
        commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());
        
        timestampLabels[currentFrame].push_back("Find cell startindices");
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, startingIndicesPass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, startingIndicesPass.m_pipelineLayout, 0, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].dispatch(workGroupCountLR, 1, 1);
        }

        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
        commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());

        timestampLabels[currentFrame].push_back("Compute density");
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, computeDensityPass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computeDensityPass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computeDensityPass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].pushConstants(computeDensityPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
            commandBuffers[currentFrame].dispatch(workGroupCountLR, 1, 1);
        }

        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
        commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());
        
        
                
        timestampLabels[currentFrame].push_back("Compute stress");
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, computeStressPass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computeStressPass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computeStressPass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].pushConstants(computeStressPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
            commandBuffers[currentFrame].dispatch(workGroupCountLR, 1, 1);
        }
        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
        commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());
        
        

        //* IISPH
        timestampLabels[currentFrame].push_back("IISPH Compute v advection");
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, iisphvAdvPass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, iisphvAdvPass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, iisphvAdvPass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].pushConstants(iisphvAdvPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
            commandBuffers[currentFrame].dispatch(workGroupCountLR, 1, 1);
        }
        
        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
        commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());
        
        
        timestampLabels[currentFrame].push_back("IISPH Compute rho advection");
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, iisphRhoAdvPass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, iisphRhoAdvPass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, iisphRhoAdvPass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].pushConstants(iisphRhoAdvPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
            commandBuffers[currentFrame].dispatch(workGroupCountLR, 1, 1);
        }
        
        //* wait for compute pass
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
        commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());
        
        
        m_core->endCommands(commandBuffers[currentFrame]);

        //Submit Commandbuffer and wait for execution to finish
        {
            std::array<vk::CommandBuffer, 1> submitComputeCommandBuffers = { 
                commandBuffers[currentFrame]
            }; 

            std::vector<vk::Semaphore> signalComputeSemaphores = {iisphSemaphores[currentFrame]};

            vk::SubmitInfo computeSubmitInfo{
                {},
                {},
                submitComputeCommandBuffers,
                signalComputeSemaphores
            };

            m_core->getDevice().resetFences(iisphFences[currentFrame]);
            m_core->getComputeQueue().submit(computeSubmitInfo, iisphFences[currentFrame]);
        
            vk::Result result = m_core->getDevice().waitForFences(iisphFences[currentFrame], VK_TRUE, UINT64_MAX);
            m_core->getDevice().resetFences(iisphFences[currentFrame]);
        }

        m_core->beginCommands(commandBuffers[currentFrame]);

        uint32_t l = 0;
        float ny = settings.maxCompression * settings.rho0;
        while ((l < 2 || std::abs(additionalData.averageDensityError) > ny) && l < 100 ) 
        {
            timestampLabels[currentFrame].push_back("IISPH Iteration " + std::to_string(l) + " Compute dijpj");
            {
                commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, iisphdijpjSolvePass.m_pipeline);
                commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, iisphdijpjSolvePass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
                commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, iisphdijpjSolvePass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
                commandBuffers[currentFrame].pushConstants(iisphdijpjSolvePass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
                commandBuffers[currentFrame].dispatch(workGroupCountLR, 1, 1);
            }
            
            //* wait for compute pass
            commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
            commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());
            
        
            timestampLabels[currentFrame].push_back("IISPH Iteration " + std::to_string(l) + " Compute pressure");
            {
                commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, iisphPressureSolvePass.m_pipeline);
                commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, iisphPressureSolvePass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
                commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, iisphPressureSolvePass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
                commandBuffers[currentFrame].pushConstants(iisphPressureSolvePass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
                commandBuffers[currentFrame].dispatch(workGroupCountLR, 1, 1);
            }
            
            //* wait for compute pass
            commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
            commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());
            
        
            timestampLabels[currentFrame].push_back("IISPH Iteration " + std::to_string(l) + " Write last pressure");
            {
                commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, iisphSolveEndPass.m_pipeline);
                commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, iisphSolveEndPass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
                commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, iisphSolveEndPass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
                commandBuffers[currentFrame].pushConstants(iisphSolveEndPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
                commandBuffers[currentFrame].dispatch(workGroupCountLR, 1, 1);
            }
            
            //* wait for compute pass
            commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);
            commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());

            m_core->endCommands(commandBuffers[currentFrame]);
            
            //Submit Commandbuffer and wait for execution to finish
            {
                std::vector<vk::Semaphore> waitSemaphores = {
                    iisphSemaphores[currentFrame]
                };
                std::vector<vk::PipelineStageFlags> waitStages = {
                    vk::PipelineStageFlagBits::eComputeShader
                };
                std::vector<vk::Semaphore> signalSemaphores = {
                    iisphSemaphores[currentFrame]
                };
                std::array<vk::CommandBuffer, 1> submitCommandBuffers = { 
                    commandBuffers[currentFrame]
                };
                vk::SubmitInfo submitInfo(waitSemaphores, waitStages, submitCommandBuffers, signalSemaphores);

                m_core->getDevice().resetFences(iisphFences[currentFrame]);
                m_core->getComputeQueue().submit(submitInfo, iisphFences[currentFrame]);

                vk::Result result = m_core->getDevice().waitForFences(iisphFences[currentFrame], VK_TRUE, UINT64_MAX);
            }

            void* mappedData = m_core->mapBuffer(additionalDataBuffer[currentFrame]);
            // Get average density error
            memcpy(&additionalData, mappedData, (size_t) sizeof(AdditionalData));
            additionalData.averageDensityError /= lrParticles.size(); // average density
            additionalData.averageDensityError -= settings.rho0; // average density error
            
            // Reset average density error
            AdditionalData resetData;
            memcpy(mappedData, &resetData, (size_t) sizeof(AdditionalData));
            m_core->flushBuffer(additionalDataBuffer[currentFrame], 0, (size_t) sizeof(AdditionalData));
            m_core->unmapBuffer(additionalDataBuffer[currentFrame]);
            // Increase iteration count
            l++;
            m_core->beginCommands(commandBuffers[currentFrame]);
            commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eHost, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);      
        }
        
        simulationMetrics.averageDensityError.append(additionalData.averageDensityError);
        simulationMetrics.iterationCount.append(l);

        timestampLabels[currentFrame].push_back("Compute pressure force");
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, computeInternalForcePass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computeInternalForcePass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, computeInternalForcePass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].pushConstants(computeInternalForcePass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
            commandBuffers[currentFrame].dispatch(workGroupCountLR, 1, 1);
        }

        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);      
        commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());


        timestampLabels[currentFrame].push_back("Integrate");
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, integratePass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, integratePass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].pushConstants(integratePass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
            commandBuffers[currentFrame].dispatch(workGroupCountLR, 1, 1);
        }

        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eComputeShader, {}, writeReadBarrier, nullptr, nullptr);      
        commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());
        
        
        timestampLabels[currentFrame].push_back("Advect HR particles");
        {
            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eCompute, advectionPass.m_pipeline);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, advectionPass.m_pipelineLayout, 0, 1, &descriptorSetsParticles[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eCompute, advectionPass.m_pipelineLayout, 1, 1, &descriptorSetsGrid[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].pushConstants(advectionPass.m_pipelineLayout, vk::ShaderStageFlagBits::eCompute, 0, sizeof(SPHSettings), &settings);
            commandBuffers[currentFrame].dispatch(workGroupCountHR, 1, 1);
        }

        // * Wait for copy action
        commandBuffers[currentFrame].pipelineBarrier(vk::PipelineStageFlagBits::eComputeShader, vk::PipelineStageFlagBits::eTransfer | vk::PipelineStageFlagBits::eComputeShader | vk::PipelineStageFlagBits::eVertexInput, {}, writeReadBarrier, nullptr, nullptr);
        commandBuffers[currentFrame].writeTimestamp(vk::PipelineStageFlagBits::eComputeShader, timeQueryPools[currentFrame], (uint32_t)timestampLabels[currentFrame].size());
        
        simulationStepForward = false;
        
    }
    else{
        m_core->endCommands(commandBuffers[currentFrame]);

        //Submit Commandbuffer and wait for execution to finish
        {
            std::array<vk::CommandBuffer, 1> submitComputeCommandBuffers = { 
                commandBuffers[currentFrame]
            }; 

            std::vector<vk::Semaphore> signalComputeSemaphores = {iisphSemaphores[currentFrame]};

            vk::SubmitInfo computeSubmitInfo{
                {},
                {},
                submitComputeCommandBuffers,
                signalComputeSemaphores
            };

            m_core->getDevice().resetFences(iisphFences[currentFrame]);
            m_core->getComputeQueue().submit(computeSubmitInfo, iisphFences[currentFrame]);
        
            vk::Result result = m_core->getDevice().waitForFences(iisphFences[currentFrame], VK_TRUE, UINT64_MAX);
            m_core->getDevice().resetFences(iisphFences[currentFrame]);
        }

        m_core->beginCommands(commandBuffers[currentFrame]);

    }
    
    m_core->endCommands(commandBuffers[currentFrame]);
}


void GranularMatter::createDescriptorPool() {

    descriptorPool = m_core->createDescriptorPool({
        { vk::DescriptorType::eStorageBuffer, (2 + 1 + 1 + 1 + 1 + 1) * gpu::MAX_FRAMES_IN_FLIGHT },
        { vk::DescriptorType::eSampler, 1 * gpu::MAX_FRAMES_IN_FLIGHT },
        { vk::DescriptorType::eSampledImage, (uint32_t)signedDistanceFieldViews.size() * gpu::MAX_FRAMES_IN_FLIGHT },
    }, (1 + 1 + 1) * gpu::MAX_FRAMES_IN_FLIGHT);
}

void GranularMatter::createDescriptorSetLayout() {
    
    descriptorSetLayoutGrid = m_core->createDescriptorSetLayout({
        {0, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute}
    });

    descriptorSetLayoutParticles = m_core->createDescriptorSetLayout({
        {1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute},
        {2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute},
        {3, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute},
        {4, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eCompute},
        {5, vk::DescriptorType::eSampler, vk::ShaderStageFlagBits::eCompute},
        {6, vk::DescriptorType::eSampledImage, (uint32_t)signedDistanceFieldViews.size(), vk::ShaderStageFlagBits::eCompute, vk::DescriptorBindingFlagBits::eVariableDescriptorCount | vk::DescriptorBindingFlagBits::ePartiallyBound }
    });

}

void GranularMatter::createDescriptorSets() {

    descriptorSetsGrid = m_core->allocateDescriptorSets(descriptorSetLayoutGrid, descriptorPool, gpu::MAX_FRAMES_IN_FLIGHT);
    descriptorSetsParticles = m_core->allocateDescriptorSets(descriptorSetLayoutParticles, descriptorPool, gpu::MAX_FRAMES_IN_FLIGHT);
    
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        m_core->addDescriptorWrite(descriptorSetsGrid[i], { 0, vk::DescriptorType::eStorageBuffer, particleCellBuffer, sizeof(ParticleGridEntry) * particleCells.size() });
        m_core->addDescriptorWrite(descriptorSetsGrid[i], { 2, vk::DescriptorType::eStorageBuffer, startingIndicesBuffers, sizeof(uint32_t) * startingIndices.size() });
        m_core->updateDescriptorSet(descriptorSetsGrid[i]);
        
        m_core->addDescriptorWrite(descriptorSetsParticles[i], { 1, vk::DescriptorType::eStorageBuffer, particlesBufferB, sizeof(LRParticle) * lrParticles.size() });
        m_core->addDescriptorWrite(descriptorSetsParticles[i], { 2, vk::DescriptorType::eStorageBuffer, particlesBufferHR, sizeof(HRParticle) * hrParticles.size() });
        m_core->addDescriptorWrite(descriptorSetsParticles[i], { 3, vk::DescriptorType::eStorageBuffer, additionalDataBuffer[i], sizeof(AdditionalData) });
        m_core->addDescriptorWrite(descriptorSetsParticles[i], { 4, vk::DescriptorType::eStorageBuffer, volumeMapTransformsBuffer, volumeMapTransforms.size() * sizeof(VolumeMapTransform)});
        m_core->addDescriptorWrite(descriptorSetsParticles[i], { 5, vk::DescriptorType::eSampler, volumeMapSampler, {}, {} });
        m_core->addDescriptorWrite(descriptorSetsParticles[i], { 6, vk::DescriptorType::eSampledImage, {}, signedDistanceFieldViews, vk::ImageLayout::eShaderReadOnlyOptimal });
        m_core->updateDescriptorSet(descriptorSetsParticles[i]);
    }
}

#define EPSILON 0.0000001f

float cubicSplineKernel(float r, float h){
    float alpha = 1.f / (4.f * (float)M_PI);
    float q = r / h;
    if (0.f <= q && q <= 1.0) {
        return alpha * (std::powf((2.f - q), 3.f) - 4.f * std::powf((1.f - q), 2.f));
    } else if (1.f <= q && q <= 2.f) {
        return alpha * std::powf((2.f - q), 3.f);
    } else {
        return 0.f;
    }
}


float cubicExtension(float r){
    float h = settings.h_LR;
    if(r < EPSILON){
        return 1;
    }
    else if(r < h){
        return cubicSplineKernel(r, h) / cubicSplineKernel(0, h);
    }
    else{
        return 0;
    }
}

// glm::vec4 adjustKernelRadiusOffset(glm::vec3 position){
//     return glm::vec4(position - 2 * settings.h_LR, 1.0);
// }
// glm::vec4 adjustKernelRadiusScale(glm::vec3 scale){
//     return glm::vec4(scale + 2 * settings.h_LR, 1.0);
// }

void GranularMatter::createSignedDistanceFields()
{

    //* Setup rigid bodies
    // float halfBoxSize = 2.f;
    // Box3D box{ 
    //     glm::vec3(halfBoxSize, halfBoxSize, halfBoxSize)
    // };
    // box.position = glm::vec3(0, halfBoxSize, 0);
    // box.scale = glm::vec3(1, 0.5, 1);
    // rigidBodies.push_back(&box);

    Plane3D floor{ glm::vec3(0, 1, 0), 0};
    // floor.position = glm::vec3(0, -9.75, 0);
    rigidBodies.push_back(&floor);

    // Plane3D wallLeft{ glm::vec3(1, 0, 0), 0};
    // wallLeft.position = glm::vec3(-settings.DOMAIN_WIDTH / 2.f, 0, 0);
    // rigidBodies.push_back(&wallLeft);

    // Plane3D wallRight{ glm::vec3(-1, 0, 0), 0};
    // wallRight.position = glm::vec3(settings.DOMAIN_WIDTH / 2.f, 0, 0);
    // rigidBodies.push_back(&wallRight);

    // Plane3D wallBack{ glm::vec3(0, 0, 1), 0};
    // wallBack.position = glm::vec3(0, 0, -settings.DOMAIN_WIDTH / 2);
    // rigidBodies.push_back(&wallBack);

    // Plane3D wallFront{ glm::vec3(0, 0, -1), 0};
    // wallFront.position = glm::vec3(0, 0, settings.DOMAIN_WIDTH / 2);
    // rigidBodies.push_back(&wallFront);

    Mesh3D hourglasRB(ASSETS_PATH"/models/dump_truck.glb");
    rigidBodies.push_back(&hourglasRB);



    glm::vec3 baseTextureSize = { 32, 32, 32 };
    std::cout << "Generating volume maps...";
    for(auto rb : rigidBodies){
        //* Extend area by kernel radius
        AABB aabb = rb->aabb;

        aabb.min -= 2.f * settings.h_LR;
        aabb.max += 2.f * settings.h_LR;
    
        auto baseSize = aabb.size();
        auto smallestDimension = std::min(baseSize.x, std::min(baseSize.y, baseSize.z));
        auto sizeRatio = baseSize / smallestDimension;
        glm::vec3 textureSize = glm::ceil(baseTextureSize * sizeRatio);
        std::cout << textureSize.x << " " << textureSize.y << " " << textureSize.z << std::endl;
     
        //* Get Sampling Step Size
        glm::vec3 stepSize =  aabb.size() / (textureSize);
        std::vector<glm::vec4> volumeMap;
        for(int z = 0; z < textureSize.z; z++){
            for(int y = 0; y < textureSize.y; y++){
                for(int x = 0; x < textureSize.x; x++){
                    glm::vec3 samplePoint = glm::vec3{
                        aabb.min.x + x * stepSize.x + (0.5 * stepSize.x), 
                        aabb.min.y + y * stepSize.y + (0.5 * stepSize.y), 
                        aabb.min.z + z * stepSize.z + (0.5 * stepSize.z)  
                    };
                    
                    float sd = rb->signedDistance(samplePoint) + settings.r_LR * 1.f;
                    float volume = cubicExtension(sd);

                    glm::vec3 nearestPoint = rb->signedDistanceGradient(samplePoint);
                    nearestPoint += glm::normalize(rb->signedDistanceGradient(samplePoint)) * settings.r_LR * 1.f;
                    volumeMap.push_back(glm::vec4(nearestPoint.x, nearestPoint.y, nearestPoint.z, volume));
                }
            }
        }
        //* create vulkan texture
        auto image = m_core->image3DFromData(volumeMap.data(), vk::ImageUsageFlagBits::eSampled, vma::MemoryUsage::eAutoPreferDevice, {}, (uint32_t)textureSize.x, (uint32_t)textureSize.y, (uint32_t)textureSize.z, vk::Format::eR32G32B32A32Sfloat);
        signedDistanceFields.push_back(image);
        auto view = m_core->createImageView3D(image, vk::Format::eR32G32B32A32Sfloat); 
        signedDistanceFieldViews.push_back(view);
        auto transform = VolumeMapTransform();
        transform.position = glm::vec4(rb->position + aabb.center(), 1.0);
        transform.scale = glm::vec4((glm::vec3(1.0) / (rb->scale * aabb.size())), 1.0);
        volumeMapTransforms.push_back(transform);
    }
    std::cout << " done." << std::endl;
    volumeMapSampler = m_core->createSampler(vk::SamplerAddressMode::eClampToEdge);

}

void GranularMatter::destroy(){
    destroyFrameResources();
    vk::Device device = m_core->getDevice();

    initPass.destroy();
    bitonicSortPass.destroy();
    startingIndicesPass.destroy();
    computeDensityPass.destroy();
    
    iisphvAdvPass.destroy();
    iisphRhoAdvPass.destroy();
    iisphdijpjSolvePass.destroy();
    iisphPressureSolvePass.destroy();
    iisphSolveEndPass.destroy();

    computeStressPass.destroy();
    computeInternalForcePass.destroy();
    integratePass.destroy();
    advectionPass.destroy();
    
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        m_core->getDevice().destroyQueryPool(timeQueryPools[i]);
    }

    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        m_core->destroyBuffer(additionalDataBuffer[i]);
        
    }

    m_core->destroyBuffer(volumeMapTransformsBuffer);
    m_core->destroyBuffer(particlesBufferB);
    m_core->destroyBuffer(particlesBufferHR);
    m_core->destroyBuffer(particleCellBuffer);
    m_core->destroyBuffer(startingIndicesBuffers);

    m_core->destroyDescriptorSetLayout(descriptorSetLayoutGrid);
    m_core->destroyDescriptorSetLayout(descriptorSetLayoutParticles);
    
    m_core->destroyDescriptorPool(descriptorPool);

    for(auto view : signedDistanceFieldViews){
        m_core->destroyImageView(view);
    }

    for(auto image : signedDistanceFields){
        m_core->destroyImage(image);
    }

    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        m_core->getDevice().destroyFence(iisphFences[i]);
        m_core->getDevice().destroySemaphore(iisphSemaphores[i]);
    }
        

    m_core->destroySampler(volumeMapSampler);
}

//Todo: mass * exp(p.position.y - 0)
//Todo: check https://gamma.cs.unc.edu/granular/narain-2010-granular.pdf
//Todo: check https://www.oofem.org/resources/doc/matlibmanual/html/node13.html
//Todo: check https://animation.rwth-aachen.de/media/papers/67/2020-TVCG-ImplicitBoundaryHandling.pdf
//Todo: check https://arxiv.org/pdf/2308.01629.pdf
//Todo: check https://ieeexplore.ieee.org/document/
//Todo: check https://dds.sciengine.com/cfs/files/pdfs/view/1674-7321/ezZupMJ5Tme8nR78c.pdf
//Todo: check Real-Time Simulation of Aeolian Sand Movement and Sand Ripple Evolution: A Method Based on the Physics of Blown Sand
//Todo: check https://cs.dartmouth.edu/~wjarosz/publications/meng15granular.html
//Todo: check https://cg.informatik.uni-freiburg.de/publications/2019_CGF_CompressedNeighbors.pdf