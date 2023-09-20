#include "core.h"
namespace gpu {
    class ComputePass{
        public:
            inline ComputePass(){};
            ComputePass(gpu::Core* core, std::string shaderFile, vk::DescriptorSetLayout descriptorSetLayout);
            inline ~ComputePass(){};

            void destroy();

            vk::Pipeline m_pipeline;
            vk::PipelineLayout m_pipelineLayout;
        private:
            gpu::Core* m_core;

            vk::ShaderModule m_shaderModule;
    };    
}
