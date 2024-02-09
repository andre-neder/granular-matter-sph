#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include "triangle_renderpass.h"
#include <chrono>
#include "global.h"

#include "initializers.h"

using namespace gpu;

    TriangleRenderPass::TriangleRenderPass(gpu::Core* core, gpu::Camera* camera){
        m_core = core;
        m_camera = camera;
    }
    void TriangleRenderPass::init(){

        vertShaderModule = m_core->loadShaderModule(SHADER_PATH"/triangle.vert");
        fragShaderModule = m_core->loadShaderModule(SHADER_PATH"/triangle.frag");

        descriptorSetLayout = m_core->createDescriptorSetLayout({
            {0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex }
        });
        createRenderPass();
        createGraphicsPipeline();
        initFrameResources();
    }

     void TriangleRenderPass::createFramebuffers(){
        framebuffers.resize(m_core->getSwapChainImageCount());
        for (int i = 0; i < m_core->getSwapChainImageCount(); i++) {
            std::array<vk::ImageView, 2> attachments = {
                m_core->getSwapChainImageView(i),
                m_core->getSwapChainDepthImageView()
            };
            framebuffers[i] = m_core->createFramebuffer(renderPass, attachments);
        }
    }


    void TriangleRenderPass::createCommandBuffers(){
        commandBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        vk::CommandBufferAllocateInfo allocInfo(m_core->getCommandPool(), vk::CommandBufferLevel::ePrimary, (uint32_t) commandBuffers.size());
        commandBuffers = m_core->getDevice().allocateCommandBuffers(allocInfo);
    }

    void TriangleRenderPass::initFrameResources(){
        createFramebuffers();
        createUniformBuffers();
        createDescriptorPool();
        createDescriptorSets();
        createCommandBuffers();
    }
    void TriangleRenderPass::createRenderPass(){
        renderPass = m_core->createColorDepthRenderPass(vk::AttachmentLoadOp::eLoad, vk::AttachmentStoreOp::eStore);
    }
    
    void TriangleRenderPass::update(int currentFrame, int imageIndex, float dt){
        updateUniformBuffer(currentFrame);

        vk::CommandBufferBeginInfo beginInfo;
        commandBuffers[currentFrame].begin(beginInfo);
        vk::ClearValue colorClear;
        colorClear.color = vk::ClearColorValue(0.1f, 0.1f, 0.1f, 1.0f);
        vk::ClearValue depthClear;
        depthClear.depthStencil = vk::ClearDepthStencilValue(1.f);
        std::array<vk::ClearValue, 2> clearValues = {
            colorClear, 
            depthClear
        };
        
        vk::RenderPassBeginInfo renderPassInfo(renderPass, framebuffers[imageIndex], vk::Rect2D({0, 0}, m_core->getSwapChainExtent()), clearValues);

        commandBuffers[currentFrame].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);
            
            vk::Viewport viewport(0.0f, 0.0f, (float)m_core->getSwapChainExtent().width, (float)m_core->getSwapChainExtent().height, 0.0f, 1.0f);
            commandBuffers[currentFrame].setViewport(0, viewport);

            vk::Rect2D scissor(vk::Offset2D(0, 0),m_core->getSwapChainExtent());
            commandBuffers[currentFrame].setScissor(0, scissor);

            commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
            
            commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);
            commandBuffers[currentFrame].pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(SPHSettings), &settings);

            for (auto&& model : models){
                std::vector<vk::Buffer> vertexBuffers = {model.vertexBuffer};
                std::vector<vk::DeviceSize> offsets = {0};
                commandBuffers[currentFrame].bindVertexBuffers(0, vertexBuffers, offsets);
                commandBuffers[currentFrame].bindIndexBuffer(model.indexBuffer, 0, vk::IndexType::eUint32);
                commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 1, 1, &model.descriptorSets[currentFrame], 0, nullptr);
                for (auto&& node : model._linearNodes)
                {
                    for (auto&& primitive : node->primitives)
                    {
                        commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 2, 1, &model.materialDescriptorSets[primitive->materialIndex][currentFrame], 0, nullptr);
                        commandBuffers[currentFrame].drawIndexed(primitive->indexCount, 1, primitive->firstIndex, 0, 0);
                    }
                }
            }
            // commandBuffers[currentFrame].drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
            commandBuffers[currentFrame].endRenderPass();
            commandBuffers[currentFrame].end();
      
    }
    
    void TriangleRenderPass::createDescriptorSets() {
     
        descriptorSets = m_core->allocateDescriptorSets(descriptorSetLayout, descriptorPool, gpu::MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {

            m_core->addDescriptorWrite(descriptorSets[i], {0, vk::DescriptorType::eUniformBuffer, uniformBuffers[i], sizeof(UniformBufferObject)});
            m_core->updateDescriptorSet(descriptorSets[i]);
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
            Model::getTexturesLayout(m_core),
            Model::getMaterialLayout(m_core)
        };
        vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, allLayouts.size(), allLayouts.data(), 1, &pushConstantRange, nullptr);

        pipelineLayout = m_core->getDevice().createPipelineLayout(pipelineLayoutInfo);

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
            renderPass
        );
        
        vk::Result result;
        std::tie(result, graphicsPipeline) = m_core->getDevice().createGraphicsPipeline( nullptr, pipelineInfo);
        switch ( result ){
            case vk::Result::eSuccess: break;
            default: throw std::runtime_error("failed to create graphics Pipeline!");
        }
    }

    void gpu::TriangleRenderPass::destroyGraphicsPipeline()
    {
        m_core->getDevice().destroyPipeline(graphicsPipeline);
    }

    void TriangleRenderPass::createUniformBuffers() {
        vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

        uniformBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            uniformBuffers[i] = m_core->createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst,  vma::MemoryUsage::eAutoPreferDevice, vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
        }

    }

    void TriangleRenderPass::createDescriptorPool() {
        descriptorPool = m_core->createDescriptorPool({
            { vk::DescriptorType::eUniformBuffer, 1 * gpu::MAX_FRAMES_IN_FLIGHT }
        }, 1 * gpu::MAX_FRAMES_IN_FLIGHT );
    }

    void TriangleRenderPass::destroyFrameResources(){
        vk::Device device = m_core->getDevice();
        for (auto framebuffer : framebuffers) {
            device.destroyFramebuffer(framebuffer);
        }
        device.freeCommandBuffers(m_core->getCommandPool(), commandBuffers);

        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            m_core->destroyBuffer(uniformBuffers[i]);
        }

        m_core->destroyDescriptorPool(descriptorPool);
    }
    void TriangleRenderPass::destroy(){
        destroyFrameResources();

        vk::Device device = m_core->getDevice();

        device.destroyShaderModule(fragShaderModule);
        device.destroyShaderModule(vertShaderModule);
        // device.destroyShaderModule(geomShaderModule);



        device.destroyPipeline(graphicsPipeline);
        device.destroyPipelineLayout(pipelineLayout);
        device.destroyRenderPass(renderPass);

        m_core->destroyDescriptorSetLayout(descriptorSetLayout);
    }
    void TriangleRenderPass::updateUniformBuffer(uint32_t currentImage) {

        UniformBufferObject ubo{};
        ubo.model = glm::scale(glm::mat4(1.0), glm::vec3(1.0));// glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        ubo.view =  m_camera->getView();//glm::lookAt(glm::vec3(0.0f, 0.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
        ubo.proj = glm::perspective(glm::radians(45.0f), m_core->getSwapChainExtent().width / (float) m_core->getSwapChainExtent().height, 0.1f, 1000.0f);
        ubo.proj[1][1] *= -1;
        

        m_core->updateBufferData(uniformBuffers[currentImage], &ubo, (size_t) sizeof(ubo));

    }
