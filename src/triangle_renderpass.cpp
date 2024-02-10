#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include "triangle_renderpass.h"
#include <chrono>
#include "global.h"

#include "initializers.h"

using namespace gpu;

    TriangleRenderPass::TriangleRenderPass(gpu::Core* core, gpu::Camera* camera){
        _core = core;
        m_camera = camera;
    }
    void TriangleRenderPass::init(){

        vertShaderModule = _core->loadShaderModule(SHADER_PATH"/triangle.vert");
        fragShaderModule = _core->loadShaderModule(SHADER_PATH"/triangle.frag");

        descriptorSetLayout = _core->createDescriptorSetLayout({
            {0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex }
        });
        _renderContext = RenderContext(_core, vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore);
        createGraphicsPipeline();
        initFrameResources();
    }

  

    void TriangleRenderPass::initFrameResources(){
        _renderContext.initFramebuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        _renderContext.initCommandBuffers();
    }

    
    void TriangleRenderPass::update(int currentFrame, int imageIndex, float dt){
        updateUniformBuffer(currentFrame);

        _renderContext.beginCommandBuffer();
        _renderContext.beginRenderPass(imageIndex);

            _renderContext.getCommandBuffer().bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
            
            _renderContext.getCommandBuffer().bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
            _renderContext.getCommandBuffer().pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(SPHSettings), &settings);

            for (auto&& model : models){
                std::vector<vk::Buffer> vertexBuffers = {model.vertexBuffer};
                std::vector<vk::DeviceSize> offsets = {0};
                _renderContext.getCommandBuffer().bindVertexBuffers(0, vertexBuffers, offsets);
                _renderContext.getCommandBuffer().bindIndexBuffer(model.indexBuffer, 0, vk::IndexType::eUint32);
                _renderContext.getCommandBuffer().bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 1, 1, &model.descriptorSets[currentFrame], 0, nullptr);
                for (auto&& node : model._linearNodes)
                {
                    for (auto&& primitive : node->primitives)
                    {
                        _renderContext.getCommandBuffer().bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 2, 1, &model.materialDescriptorSets[primitive->materialIndex][currentFrame], 0, nullptr);
                        _renderContext.getCommandBuffer().drawIndexed(primitive->indexCount, 1, primitive->firstIndex, 0, 0);
                    }
                }
            }
            _renderContext.endRenderPass();
            _renderContext.endCommandBuffer();
      
    }
    
    void TriangleRenderPass::createDescriptorSets() {
     
        descriptorSets = _core->allocateDescriptorSets(descriptorSetLayout, descriptorPool, gpu::MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {

            _core->addDescriptorWrite(descriptorSets[i], {0, vk::DescriptorType::eUniformBuffer, uniformBuffers[i], sizeof(UniformBufferObject)});
            _core->updateDescriptorSet(descriptorSets[i]);
        }
    }
    
    void TriangleRenderPass::createGraphicsPipeline(bool wireframe){
        vk::PipelineShaderStageCreateInfo vertShaderStageInfo({}, vk::ShaderStageFlagBits::eVertex, vertShaderModule, "main");
        // vk::PipelineShaderStageCreateInfo geomShaderStageInfo({}, vk::ShaderStageFlagBits::eGeometry, geomShaderModule, "main");
        vk::PipelineShaderStageCreateInfo fragShaderStageInfo({}, vk::ShaderStageFlagBits::eFragment, fragShaderModule, "main");

        std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
            vertShaderStageInfo, 
            // geomShaderStageInfo,
            fragShaderStageInfo
        }; 

        auto vertexDescription = Vertex::get_vertex_description();
        std::vector<vk::VertexInputBindingDescription> bindings;
        bindings.insert(bindings.end(), vertexDescription.bindings.begin(), vertexDescription.bindings.end());
        std::vector<vk::VertexInputAttributeDescription> attributes;
        attributes.insert(attributes.end(), vertexDescription.attributes.begin(), vertexDescription.attributes.end());

        vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, bindings, attributes);

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);
        vk::PipelineRasterizationStateCreateInfo rasterizer({}, VK_FALSE, VK_FALSE, wireframe ? vk::PolygonMode::eLine : vk::PolygonMode::eFill, vk::CullModeFlagBits::eNone, vk::FrontFace::eCounterClockwise, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f);
        vk::PushConstantRange pushConstantRange{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(SPHSettings)};
        std::vector<vk::DescriptorSetLayout> allLayouts = {
            descriptorSetLayout,
            Model::getTexturesLayout(_core),
            Model::getMaterialLayout(_core)
        };
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, (uint32_t)allLayouts.size(), allLayouts.data(), 1, &pushConstantRange, nullptr);

        pipelineLayout = _core->getDevice().createPipelineLayout(pipelineLayoutInfo);

        vk::GraphicsPipelineCreateInfo pipelineInfo(
            {}, 
            shaderStages, 
            &vertexInputInfo, 
            &inputAssembly, 
            {}, 
            &gpu::Initializers::DynamicViewportState, 
            &rasterizer, 
            &gpu::Initializers::DefaultMultisampleStateCreateInfo, 
            &gpu::Initializers::DefaultDepthStenilStateCreateInfo, 
            &gpu::Initializers::DefaultColorBlendStateCreateInfo, 
            &gpu::Initializers::DefaultDynamicStateCreateInfo, 
            pipelineLayout, 
            _renderContext.getRenderPass()
        );
        
        vk::Result result;
        std::tie(result, graphicsPipeline) = _core->getDevice().createGraphicsPipeline( nullptr, pipelineInfo);
        switch ( result ){
            case vk::Result::eSuccess: break;
            default: throw std::runtime_error("failed to create graphics Pipeline!");
        }
    }

    void gpu::TriangleRenderPass::destroyGraphicsPipeline()
    {
        _core->getDevice().destroyPipeline(graphicsPipeline);
    }

    void TriangleRenderPass::createUniformBuffers() {
        uniformBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            uniformBuffers[i] = _core->createBuffer(sizeof(UniformBufferObject), vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,  vma::MemoryUsage::eAutoPreferDevice, vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
        }

    }

    void TriangleRenderPass::createDescriptorPool() {
        descriptorPool = _core->createDescriptorPool({
            { vk::DescriptorType::eUniformBuffer, 1 * gpu::MAX_FRAMES_IN_FLIGHT }
        }, 1 * gpu::MAX_FRAMES_IN_FLIGHT );
    }

    void TriangleRenderPass::destroyFrameResources(){
        _renderContext.destroyFramebuffers();

        vk::Device device = _core->getDevice();
        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            _core->destroyBuffer(uniformBuffers[i]);
        }

        _core->destroyDescriptorPool(descriptorPool);
    }
    void TriangleRenderPass::destroy(){
        destroyFrameResources();

        vk::Device device = _core->getDevice();

        device.destroyShaderModule(fragShaderModule);
        device.destroyShaderModule(vertShaderModule);
        // device.destroyShaderModule(geomShaderModule);

        _renderContext.freeCommandBuffers();

        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        _renderContext.destroyRenderPass();

        _core->destroyDescriptorSetLayout(descriptorSetLayout);
    }
    void TriangleRenderPass::updateUniformBuffer(uint32_t currentImage) {

        UniformBufferObject ubo{};
        ubo.model = glm::scale(glm::mat4(1.0), glm::vec3(1.0));// glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view =  m_camera->getView();//glm::lookAt(glm::vec3(0.0f, 0.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), _core->getSwapchainExtent().width / (float) _core->getSwapchainExtent().height, 0.1f, 1000.0f);
        ubo.proj[1][1] *= -1;
        

        _core->updateBufferData(uniformBuffers[currentImage], &ubo, (size_t) sizeof(ubo));

    }
