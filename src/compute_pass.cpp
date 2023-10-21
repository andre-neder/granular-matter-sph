#include "compute_pass.h"

using namespace gpu;


ComputePass::ComputePass(gpu::Core *core, std::string shaderFile, std::vector<vk::DescriptorSetLayout> descriptorSetLayouts) : 
    ComputePass::ComputePass(core, shaderFile, descriptorSetLayouts, std::vector<gpu::SpecializationConstant>(), 0)
{
    // ComputePass(core, shaderFile, descriptorSetLayouts, std::vector<gpu::SpecializationConstant>());
}
ComputePass::ComputePass(gpu::Core *core, std::string shaderFile, std::vector<vk::DescriptorSetLayout> descriptorSetLayouts, std::vector<gpu::SpecializationConstant> specializations, uint32_t pushConstantSize)
{
    m_core = core;
    m_shaderModule = core->loadShaderModule(shaderFile);

    vk::Result result;

    std::vector<vk::SpecializationMapEntry> entries;
    std::vector<vk::SpecializationMapEntry> data;
    uint32_t offset = 0;
    uint32_t sizeOfConstant = sizeof(int32_t);
    for (auto spec : specializations)
    {
        vk::SpecializationMapEntry entry = { spec.id, offset, sizeOfConstant };
        entries.push_back(entry);
        data.push_back(spec.value);
        offset += sizeOfConstant;
    }

    // vk::SpecializationMapEntry entry = { 0, 0, sizeof(int32_t) };

    vk::PipelineShaderStageCreateInfo stageInfo;
    if(entries.size() > 0){
        vk::SpecializationInfo spec_info = {
            (uint32_t)entries.size(),
            entries.data(),
            sizeOfConstant * data.size(),
            data.data()
        };
        stageInfo = {
            {},
            vk::ShaderStageFlagBits::eCompute,
            m_shaderModule,
            "main",
            &spec_info
        };
    }
    else{
        stageInfo = {
            {},
            vk::ShaderStageFlagBits::eCompute,
            m_shaderModule,
            "main"
        };
    }
    vk::PipelineLayoutCreateInfo layoutInfo;
    // uint32_t pushConstantSize = sizeof(float) * 2;
    if(pushConstantSize > 0){
        vk::PushConstantRange pushConstantRange{
            vk::ShaderStageFlagBits::eCompute,
            0,
            pushConstantSize
        };

        layoutInfo = {
            {},
            (uint32_t)descriptorSetLayouts.size(),
            descriptorSetLayouts.data(),
            1,
            &pushConstantRange,
            nullptr
        };
    }
    else{
        layoutInfo = {
            {},
            (uint32_t)descriptorSetLayouts.size(),
            descriptorSetLayouts.data(),
            0,
            {},
            nullptr
        };
    }

    try{
        m_pipelineLayout = core->getDevice().createPipelineLayout(layoutInfo, nullptr);
    }catch(std::exception& e) {
        std::cerr << "Exception Thrown: " << e.what();
    }

    vk::ComputePipelineCreateInfo pipelineInfo{
        {},
        stageInfo,
        m_pipelineLayout,
    };

    
    std::tie(result, m_pipeline) = core->getDevice().createComputePipeline( nullptr, pipelineInfo);
    switch ( result ){
        case vk::Result::eSuccess: break;
        default: throw std::runtime_error("Failed to create compute Pipeline!");
    }
}

void ComputePass::destroy()
{
    m_core->getDevice().destroyShaderModule(m_shaderModule);
    m_core->getDevice().destroyPipelineLayout(m_pipelineLayout);
    m_core->getDevice().destroyPipeline(m_pipeline);
}
