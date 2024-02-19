#pragma once
#include <glm/glm.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include <tiny_gltf.h>
#include "camera.h"
#include "model.h"
#include "core.h"
#include "render_context.h"
#include <meshoptimizer.h>

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

            vk::Buffer vertexBuffer;
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
            std::vector<meshopt_Meshlet> _meshlets;
            std::vector<meshopt_Bounds> _meshletBounds;
            vk::Buffer _meshletBuffer;
            vk::Buffer _meshletBoundsBuffer;
            vk::Buffer _meshletVerticesBuffer;
            vk::Buffer _meshletTrianglesBuffer;
            std::vector<unsigned int> _meshletVertices;
            std::vector<unsigned char> _meshletTriangles;
   
            std::vector<vk::Buffer> uniformBuffers;
            std::vector<vk::Buffer> uniformBuffersSettings;

            vk::DescriptorPool descriptorPool;
            std::vector<vk::DescriptorSet> descriptorSets;
            vk::PipelineLayout pipelineLayout;
            vk::DescriptorSetLayout descriptorSetLayout;
            vk::ShaderModule meshShaderModule;
            vk::ShaderModule taskShaderModule;
            vk::ShaderModule fragShaderModule;
            vk::ShaderModule geomShaderModule;

            void createDescriptorSets();
            void createGraphicsPipeline();
            void createDescriptorPool();
    };
}
