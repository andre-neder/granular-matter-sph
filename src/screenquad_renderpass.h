#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include <tiny_gltf.h>
#include "renderpass.h"

struct Vertex {
    glm::vec2 pos;
    glm::vec3 color;
    glm::vec2 texCoord;

    static std::array<vk::VertexInputBindingDescription, 1> getBindingDescription() {
        std::array<vk::VertexInputBindingDescription, 1> bindingDescriptions = {
            vk::VertexInputBindingDescription(0, sizeof(Vertex), vk::VertexInputRate::eVertex)
        };
        return bindingDescriptions;
    }
    static std::array<vk::VertexInputAttributeDescription, 1> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 1> attributeDescriptions{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, pos))
        };
        return attributeDescriptions;
    }
};

const std::vector<Vertex> vertices = {
    {{-1.f, -1.f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
    {{1.f, -1.f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
    {{1.f, 1.f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
    {{-1.f, 1.f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}}
};

const std::vector<uint32_t> indices = {
    0, 1, 2, 2, 3, 0
};

namespace gpu{
    class ScreenQuadRenderPass : public RenderPass{
        public:
            ScreenQuadRenderPass(){};
            ScreenQuadRenderPass(gpu::Core* core);
            ~ScreenQuadRenderPass(){};

            void initFrameResources();
            void update(int currentFrame, int imageIndex);
            void destroyFrameResources();
            void destroy(); 
            void init();


        private:

        
   
            std::vector<vk::Buffer> vertexBuffer;
            vk::Buffer indexBuffer;
            vk::PipelineLayout pipelineLayout;
            vk::ShaderModule vertShaderModule;
            vk::ShaderModule fragShaderModule;

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
