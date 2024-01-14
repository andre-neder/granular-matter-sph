#pragma once
#include <stb_image.h>
#include <stb_image_write.h>
#include <tiny_gltf.h>
#include "renderpass.h"

#include "global.h"
#include "camera.h"
#include "model.h"

struct TriangleVertex {
    glm::vec4 pos;

    static std::array<vk::VertexInputBindingDescription, 1> getBindingDescription() {
        std::array<vk::VertexInputBindingDescription, 1> bindingDescriptions = {
            vk::VertexInputBindingDescription(0, sizeof(TriangleVertex), vk::VertexInputRate::eVertex)
        };
        return bindingDescriptions;
    }
    static std::array<vk::VertexInputAttributeDescription, 1> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 1> attributeDescriptions{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32A32Sfloat, offsetof(TriangleVertex, pos)),
        };
        return attributeDescriptions;
    }
};

const float halfBoxSize = settings.DOMAIN_HEIGHT / 4.f;
const float halfDomainWidth = settings.DOMAIN_WIDTH / 2.f;
const std::vector<TriangleVertex> vertices = {
    {{-halfDomainWidth, 0.f,-halfDomainWidth, 1.f}},
    {{halfDomainWidth, 0.f, -halfDomainWidth, 1.f}},
    {{-halfDomainWidth, 0.f, halfDomainWidth, 1.f}},
    {{halfDomainWidth, 0.f, halfDomainWidth, 1.f}},
    // {{(settings.DOMAIN_WIDTH / 2 - halfBoxSize) , 0 , 0.f, 0.f}},
    // {{(settings.DOMAIN_WIDTH / 2 + halfBoxSize) , 0 , 0.f, 0.f}},
    // {{(settings.DOMAIN_WIDTH / 2 - halfBoxSize) , halfBoxSize * 2 , 0.f, 0.f}},
    // {{(settings.DOMAIN_WIDTH / 2 + halfBoxSize) , halfBoxSize * 2 , 0.f, 0.f}},
};

const std::vector<uint32_t> indices = {
    0, 1, 2, 2, 1, 3// floor
    // 0, 2, // left
    // 1, 3, // right

    // 4, 5, //box
    // 4, 6,
    // 5, 7,
    // 6, 7
};

namespace gpu{
    class TriangleRenderPass : public RenderPass{
        public:
            TriangleRenderPass(){};
            TriangleRenderPass(gpu::Core* core, gpu::Camera* camera);
            ~TriangleRenderPass(){};

            void createFramebuffers();
            void initFrameResources();
            void update(int currentFrame, int imageIndex, float dt);
            void destroyFrameResources();
            void destroy(); 
            void init();

            std::vector<Model> models;
        private:
            gpu::Camera* m_camera;

            // Model hourglassModel;

            // std::vector<vk::Buffer> vertexBuffer;
            // vk::Buffer vertexBuffer;
            // vk::Buffer indexBuffer;
            std::vector<vk::Buffer> uniformBuffers;
            std::vector<vk::Buffer> uniformBuffersSettings;

            vk::DescriptorPool descriptorPool;
            std::vector<vk::DescriptorSet> descriptorSets;
            vk::DescriptorSetLayout descriptorSetLayout;

            vk::PipelineLayout pipelineLayout;
            vk::ShaderModule vertShaderModule;
            vk::ShaderModule fragShaderModule;
            // vk::ShaderModule geomShaderModule;

            void createRenderPass();
            void createDescriptorSets();
            void createGraphicsPipeline();
            void createDescriptorSetLayout();
            void updateUniformBuffer(uint32_t currentImage);
            void createUniformBuffers();
            void createDescriptorPool();
    };
}
