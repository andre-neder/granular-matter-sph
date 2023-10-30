#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include <tiny_gltf.h>
#include "renderpass.h"

#include "global.h"

struct LineVertex {
    glm::vec2 pos;

    static std::array<vk::VertexInputBindingDescription, 1> getBindingDescription() {
        std::array<vk::VertexInputBindingDescription, 1> bindingDescriptions = {
            vk::VertexInputBindingDescription(0, sizeof(LineVertex), vk::VertexInputRate::eVertex)
        };
        return bindingDescriptions;
    }
    static std::array<vk::VertexInputAttributeDescription, 1> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 1> attributeDescriptions{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(LineVertex, pos)),
        };
        return attributeDescriptions;
    }
};


const std::vector<LineVertex> vertices = {
    {{0.f, 0.f}},
    {{settings.DOMAIN_WIDTH, 0.f}}
};

const std::vector<uint32_t> indices = {
    0, 1
};

namespace gpu{
    class LineRenderPass : public RenderPass{
        public:
            LineRenderPass(){};
            LineRenderPass(gpu::Core* core);
            ~LineRenderPass(){};

            void initFrameResources();
            void update(int currentFrame, int imageIndex);
            void destroyFrameResources();
            void destroy(); 
            void init();

        private:


            std::vector<vk::Buffer> vertexBuffer;
            vk::Buffer indexBuffer;
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
