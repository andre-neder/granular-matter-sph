#include "core.h"
namespace gpu {
    struct SpecializationConstant{
        uint32_t id;
        uint32_t value;
        inline SpecializationConstant(uint32_t id, uint32_t value) : id(id), value(value) {};
    };

    class ComputePass{
        public:
            inline ComputePass(){};
            ComputePass(gpu::Core* core, std::string shaderFile, std::vector<vk::DescriptorSetLayout> descriptorSetLayouts);
            ComputePass(gpu::Core* core, std::string shaderFile, std::vector<vk::DescriptorSetLayout> descriptorSetLayouts, std::vector<gpu::SpecializationConstant> specializations);
            inline ~ComputePass(){};

            void destroy();

            vk::Pipeline m_pipeline;
            vk::PipelineLayout m_pipelineLayout;
        private:
            gpu::Core* m_core;

            vk::ShaderModule m_shaderModule;
    };    
}
