#include "compute_pass.h"

using namespace gpu;

ComputePass::ComputePass(gpu::Core *core, std::string shaderFile, vk::DescriptorSetLayout descriptorSetLayout)
{
    m_core = core;
    m_shaderModule = core->loadShaderModule(shaderFile);

    vk::Result result;

    vk::PipelineShaderStageCreateInfo stageInfo{
        {},
        vk::ShaderStageFlagBits::eCompute,
        m_shaderModule,
        "main"
    };

    vk::PipelineLayoutCreateInfo layoutInfo{
        {},
        descriptorSetLayout,
        {}
    };

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
