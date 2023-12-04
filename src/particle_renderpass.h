#pragma once
#include <glm/glm.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include <tiny_gltf.h>
#include "renderpass.h"
#include "camera.h"
#include "model.h"


namespace gpu{
    class ParticleRenderPass : public RenderPass{
        public:
            ParticleRenderPass(){};
            ParticleRenderPass(gpu::Core* core, gpu::Camera* camera);
            ~ParticleRenderPass(){};

            void createFramebuffers();
            void initFrameResources();
            void update(int currentFrame, int imageIndex, float dt);
            void destroyFrameResources();
            void destroy(); 
            void init();

            std::vector<vk::Buffer> vertexBuffer;
            uint32_t vertexCount;

            std::array<vk::VertexInputBindingDescription, 1> bindingDescription;
            std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions;

        private:
            gpu::Camera* m_camera;

            Model particleModel;
            vk::Buffer particleModelIndexBuffer;
            vk::Buffer particleModelVertexBuffer;
   
            std::vector<vk::Buffer> uniformBuffers;
            std::vector<vk::Buffer> uniformBuffersSettings;
            // vk::Image textureImage;
            // vk::ImageView textureImageView;
            // vk::Sampler textureSampler;
            vk::DescriptorPool descriptorPool;
            std::vector<vk::DescriptorSet> descriptorSets;
            vk::PipelineLayout pipelineLayout;
            vk::DescriptorSetLayout descriptorSetLayout;
            vk::ShaderModule vertShaderModule;
            vk::ShaderModule fragShaderModule;
            vk::ShaderModule geomShaderModule;

            void createRenderPass();
            void createDescriptorSets();
            void createGraphicsPipeline();
            void createTextureImage();
            void createDescriptorSetLayout();
            void updateUniformBuffer(uint32_t currentImage);
            void createUniformBuffers();
            void createDescriptorPool();
    };
}
