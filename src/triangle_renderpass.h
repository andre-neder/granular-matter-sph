#pragma once
#include <stb_image.h>
#include <stb_image_write.h>
#include <tiny_gltf.h>
#include "core.h"
#include "global.h"
#include "camera.h"
#include "model.h"
#include "render_context.h"

namespace gpu{
    class TriangleRenderPass{
        public:
            TriangleRenderPass(){};
            TriangleRenderPass(gpu::Core* core, gpu::Camera* camera);
            ~TriangleRenderPass(){};

            void initFrameResources();
            void update(int currentFrame, int imageIndex, float dt);
            void destroyFrameResources();
            void destroy(); 
            void init();

            inline vk::CommandBuffer getCommandBuffer(int index){ return _renderContext.getCommandBuffer(); };

            void createGraphicsPipeline(bool wireframe = false);
            void destroyGraphicsPipeline();

            std::vector<Model> models;
        private:
            gpu::Camera* m_camera;
            gpu::Core* _core;

            vk::Pipeline graphicsPipeline;
            
            RenderContext _renderContext;

            std::vector<vk::Buffer> uniformBuffers;
            std::vector<vk::Buffer> uniformBuffersSettings;

            vk::DescriptorPool descriptorPool;
            std::vector<vk::DescriptorSet> descriptorSets;
            vk::DescriptorSetLayout descriptorSetLayout;

            vk::PipelineLayout pipelineLayout;
            vk::ShaderModule vertShaderModule;
            vk::ShaderModule fragShaderModule;
            // vk::ShaderModule geomShaderModule;

            void createDescriptorSets();
            
            void updateUniformBuffer(uint32_t currentImage);
            void createUniformBuffers();
            void createDescriptorPool();
    };
}
