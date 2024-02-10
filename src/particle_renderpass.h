#pragma once
#include <glm/glm.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include <tiny_gltf.h>
#include "camera.h"
#include "model.h"
#include "core.h"
#include "render_context.h"


namespace gpu{
    class ParticleRenderPass{
        public:
            ParticleRenderPass(){};
            ParticleRenderPass(gpu::Core* core, gpu::Camera* camera);
            ~ParticleRenderPass(){};

            void initFrameResources();
            void update(int imageIndex, float dt);
            void destroyFrameResources();
            void destroy(); 
            void init();

            inline vk::CommandBuffer getCommandBuffer(){ return _renderContext.getCommandBuffer(); };

            std::vector<vk::Buffer> vertexBuffer;
            uint32_t vertexCount;

            std::array<vk::VertexInputBindingDescription, 1> bindingDescription;
            std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions;

        private:
            gpu::Camera* m_camera;
            gpu::Core* _core;

            vk::Pipeline graphicsPipeline;

            RenderContext _renderContext;

            Model particleModel;
            vk::Buffer particleModelIndexBuffer;
            vk::Buffer particleModelVertexBuffer;
   
            std::vector<vk::Buffer> uniformBuffers;
            std::vector<vk::Buffer> uniformBuffersSettings;

            vk::DescriptorPool descriptorPool;
            std::vector<vk::DescriptorSet> descriptorSets;
            vk::PipelineLayout pipelineLayout;
            vk::DescriptorSetLayout descriptorSetLayout;
            vk::ShaderModule vertShaderModule;
            vk::ShaderModule fragShaderModule;
            vk::ShaderModule geomShaderModule;

            void createDescriptorSets();
            void createGraphicsPipeline();
            void updateUniformBuffer(uint32_t currentImage);
            void createUniformBuffers();
            void createDescriptorPool();
    };
}
