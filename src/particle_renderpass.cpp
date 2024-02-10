#include "particle_renderpass.h"
#include <chrono>
#include "global.h"
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include "initializers.h"

using namespace gpu;

ParticleRenderPass::ParticleRenderPass(gpu::Core *core, gpu::Camera *camera)
{
    _core = core;
    m_camera = camera;

    particleModel = Model();
    // particleModel.load_from_glb(ASSETS_PATH "/models/sphere.glb");
    particleModel.load_from_glb(ASSETS_PATH "/models/grain_smooth.glb");
    particleModelIndexBuffer = _core->bufferFromData(particleModel._indices.data(), particleModel._indices.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eIndexBuffer, vma::MemoryUsage::eAutoPreferDevice);
    particleModelVertexBuffer = _core->bufferFromData(particleModel._vertices.data(), particleModel._vertices.size() * sizeof(Vertex), vk::BufferUsageFlagBits::eVertexBuffer, vma::MemoryUsage::eAutoPreferDevice);
}
void ParticleRenderPass::init()
{

    vertShaderModule = _core->loadShaderModule(SHADER_PATH "/shader.vert");
    fragShaderModule = _core->loadShaderModule(SHADER_PATH "/shader.frag");
    geomShaderModule = _core->loadShaderModule(SHADER_PATH"/shader.geom");

    descriptorSetLayout = _core->createDescriptorSetLayout({
        {0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eGeometry}
    });
    
    _renderContext = RenderContext(_core, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore);
    createGraphicsPipeline();


    initFrameResources();
}
void ParticleRenderPass::initFrameResources()
{
    _renderContext.initFramebuffers();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    _renderContext.initCommandBuffers();
}

void ParticleRenderPass::update(int currentFrame, int imageIndex, float dt)
{
    updateUniformBuffer(currentFrame);

    _renderContext.beginCommandBuffer();
    _renderContext.beginRenderPass(imageIndex);

    _renderContext.getCommandBuffer().bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
    std::vector<vk::Buffer> vertexBuffers = {vertexBuffer[currentFrame]};
    std::vector<vk::DeviceSize> offsets = {0};

    _renderContext.getCommandBuffer().bindVertexBuffers(0, particleModelVertexBuffer, offsets);
    _renderContext.getCommandBuffer().bindIndexBuffer(particleModelIndexBuffer, 0, vk::IndexType::eUint32);

    _renderContext.getCommandBuffer().bindVertexBuffers(1, vertexBuffers, offsets);

    _renderContext.getCommandBuffer().bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

    _renderContext.getCommandBuffer().pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eGeometry, 0, sizeof(SPHSettings), &settings);

    for (auto node : particleModel._linearNodes)
    {
        for (auto primitive : node->primitives)
        {
            _renderContext.getCommandBuffer().drawIndexed(primitive->indexCount, vertexCount, primitive->firstIndex, 0, 0);
        }
    }

    _renderContext.endRenderPass();
    _renderContext.endCommandBuffer();

}

void ParticleRenderPass::createDescriptorSets()
{

    descriptorSets = _core->allocateDescriptorSets(descriptorSetLayout, descriptorPool, gpu::MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++)
    {
        _core->addDescriptorWrite(descriptorSets[i], {0, vk::DescriptorType::eUniformBuffer, uniformBuffers[i], sizeof(UniformBufferObject)});
        _core->updateDescriptorSet(descriptorSets[i]);
    }
}

void ParticleRenderPass::createGraphicsPipeline()
{
    vk::PipelineShaderStageCreateInfo vertShaderStageInfo({}, vk::ShaderStageFlagBits::eVertex, vertShaderModule, "main");
    vk::PipelineShaderStageCreateInfo geomShaderStageInfo({}, vk::ShaderStageFlagBits::eGeometry, geomShaderModule, "main");
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo({}, vk::ShaderStageFlagBits::eFragment, fragShaderModule, "main");

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
        vertShaderStageInfo, 
        // geomShaderStageInfo , 
        fragShaderStageInfo
    }; 

    auto vertexDescription = Vertex::get_vertex_description();

    std::vector<vk::VertexInputBindingDescription> bindings;
    bindings.insert(bindings.end(), vertexDescription.bindings.begin(), vertexDescription.bindings.end());
    bindings.insert(bindings.end(), bindingDescription.begin(), bindingDescription.end());
    std::vector<vk::VertexInputAttributeDescription> attributes;
    attributes.insert(attributes.end(), vertexDescription.attributes.begin(), vertexDescription.attributes.end());
    attributes.insert(attributes.end(), attributeDescriptions.begin(), attributeDescriptions.end());

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, bindings, attributes);
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);
    vk::PipelineRasterizationStateCreateInfo rasterizer({}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f);
    vk::PushConstantRange pushConstantRange{vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eGeometry, 0, sizeof(SPHSettings)};
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo({}, 1, &descriptorSetLayout, 1, &pushConstantRange, nullptr);

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
    std::tie(result, graphicsPipeline) = _core->getDevice().createGraphicsPipeline(nullptr, pipelineInfo);
    switch (result)
    {
    case vk::Result::eSuccess:
        break;
    default:
        throw std::runtime_error("failed to create graphics Pipeline!");
    }
}

void ParticleRenderPass::createUniformBuffers()
{
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);

    uniformBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++)
    {
        uniformBuffers[i] = _core->createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAutoPreferDevice, vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
    }
}

void ParticleRenderPass::createDescriptorPool()
{
    descriptorPool = _core->createDescriptorPool({{vk::DescriptorType::eUniformBuffer, 1 * gpu::MAX_FRAMES_IN_FLIGHT}}, 1 * gpu::MAX_FRAMES_IN_FLIGHT);
}

void ParticleRenderPass::destroyFrameResources()
{
    _renderContext.destroyFramebuffers();
    _renderContext.freeCommandBuffers();

    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++)
    {
        _core->destroyBuffer(uniformBuffers[i]);
    }

    _core->destroyDescriptorPool(descriptorPool);
}
void ParticleRenderPass::destroy()
{
    destroyFrameResources();

    _core->getDevice().destroyShaderModule(fragShaderModule);
    _core->getDevice().destroyShaderModule(vertShaderModule);
    _core->getDevice().destroyShaderModule(geomShaderModule);

    particleModel.destroy();
    _core->destroyBuffer(particleModelIndexBuffer);
    _core->destroyBuffer(particleModelVertexBuffer);
        
    _core->getDevice().destroyPipelineLayout(pipelineLayout);
    _core->getDevice().destroyPipeline(graphicsPipeline);
    _renderContext.destroyRenderPass();

    _core->destroyDescriptorSetLayout(descriptorSetLayout);
}
void ParticleRenderPass::updateUniformBuffer(uint32_t currentImage)
{

    UniformBufferObject ubo{};
    ubo.model = glm::mat4(1.0f);
    ubo.view = m_camera->getView();
    ubo.proj = glm::perspective(glm::radians(45.0f), _core->getSwapChainExtent().width / (float)_core->getSwapChainExtent().height, 0.1f, 1000.0f);
    ubo.proj[1][1] *= -1;

    _core->updateBufferData(uniformBuffers[currentImage], &ubo, (size_t) sizeof(ubo));
}
