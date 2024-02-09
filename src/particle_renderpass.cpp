#include "particle_renderpass.h"
#include <chrono>
#include "global.h"
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include "initializers.h"

using namespace gpu;

ParticleRenderPass::ParticleRenderPass(gpu::Core *core, gpu::Camera *camera)
{
    m_core = core;
    m_camera = camera;

    particleModel = Model();
    // particleModel.load_from_glb(ASSETS_PATH "/models/sphere.glb");
    particleModel.load_from_glb(ASSETS_PATH "/models/grain_smooth.glb");
    particleModelIndexBuffer = m_core->bufferFromData(particleModel._indices.data(), particleModel._indices.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eIndexBuffer, vma::MemoryUsage::eAutoPreferDevice);
    particleModelVertexBuffer = m_core->bufferFromData(particleModel._vertices.data(), particleModel._vertices.size() * sizeof(Vertex), vk::BufferUsageFlagBits::eVertexBuffer, vma::MemoryUsage::eAutoPreferDevice);
}
void ParticleRenderPass::init()
{

    vertShaderModule = m_core->loadShaderModule(SHADER_PATH "/shader.vert");
    fragShaderModule = m_core->loadShaderModule(SHADER_PATH "/shader.frag");
    geomShaderModule = m_core->loadShaderModule(SHADER_PATH"/shader.geom");

    descriptorSetLayout = m_core->createDescriptorSetLayout({
        {0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eGeometry}
    });
    createRenderPass();
    createGraphicsPipeline();


    initFrameResources();
}
void ParticleRenderPass::initFrameResources()
{
    createFramebuffers();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
}

void ParticleRenderPass::createFramebuffers()
{
    framebuffers.resize(m_core->getSwapChainImageCount());
    for (int i = 0; i < m_core->getSwapChainImageCount(); i++)
    {
        std::array<vk::ImageView, 2> attachments = {
            m_core->getSwapChainImageView(i),
            m_core->getSwapChainDepthImageView()
        };
        framebuffers[i] = m_core->createFramebuffer(renderPass, attachments);
    }
}
void ParticleRenderPass::createCommandBuffers(){
    commandBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    vk::CommandBufferAllocateInfo allocInfo(m_core->getCommandPool(), vk::CommandBufferLevel::ePrimary, (uint32_t) commandBuffers.size());
    commandBuffers = m_core->getDevice().allocateCommandBuffers(allocInfo);
}

void ParticleRenderPass::createRenderPass()
{
    renderPass = m_core->createColorDepthRenderPass(vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore);
}

void ParticleRenderPass::update(int currentFrame, int imageIndex, float dt)
{
    updateUniformBuffer(currentFrame);

    vk::CommandBufferBeginInfo beginInfo;
    commandBuffers[currentFrame].begin(beginInfo);
  
    std::array<vk::ClearValue, 2> clearValues = {
        vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}),
        vk::ClearDepthStencilValue({1.0f, 0})
    };
    vk::RenderPassBeginInfo renderPassInfo(renderPass, framebuffers[imageIndex], vk::Rect2D({0, 0}, m_core->getSwapChainExtent()), clearValues);

    commandBuffers[currentFrame].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

    vk::Viewport viewport(0.0f, 0.0f, (float)m_core->getSwapChainExtent().width, (float)m_core->getSwapChainExtent().height, 0.0f, 1.0f);
    commandBuffers[currentFrame].setViewport(0, viewport);

    vk::Rect2D scissor(vk::Offset2D(0, 0), m_core->getSwapChainExtent());
    commandBuffers[currentFrame].setScissor(0, scissor);

    commandBuffers[currentFrame].bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);
    std::vector<vk::Buffer> vertexBuffers = {vertexBuffer[currentFrame]};
    std::vector<vk::DeviceSize> offsets = {0};

    commandBuffers[currentFrame].bindVertexBuffers(0, particleModelVertexBuffer, offsets);
    commandBuffers[currentFrame].bindIndexBuffer(particleModelIndexBuffer, 0, vk::IndexType::eUint32);

    commandBuffers[currentFrame].bindVertexBuffers(1, vertexBuffers, offsets);

    commandBuffers[currentFrame].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[currentFrame], 0, nullptr);

    commandBuffers[currentFrame].pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eGeometry, 0, sizeof(SPHSettings), &settings);

    for (auto node : particleModel._linearNodes)
    {
        for (auto primitive : node->primitives)
        {
            commandBuffers[currentFrame].drawIndexed(primitive->indexCount, vertexCount, primitive->firstIndex, 0, 0);
        }
    }

    commandBuffers[currentFrame].endRenderPass();
    commandBuffers[currentFrame].end();

}

void ParticleRenderPass::createDescriptorSets()
{

    descriptorSets = m_core->allocateDescriptorSets(descriptorSetLayout, descriptorPool, gpu::MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++)
    {
        m_core->addDescriptorWrite(descriptorSets[i], {0, vk::DescriptorType::eUniformBuffer, uniformBuffers[i], sizeof(UniformBufferObject)});
        m_core->updateDescriptorSet(descriptorSets[i]);
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
    std::tie(result, graphicsPipeline) = m_core->getDevice().createGraphicsPipeline(nullptr, pipelineInfo);
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
        uniformBuffers[i] = m_core->createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAutoPreferDevice, vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
    }
}

void ParticleRenderPass::createDescriptorPool()
{
    descriptorPool = m_core->createDescriptorPool({{vk::DescriptorType::eUniformBuffer, 1 * gpu::MAX_FRAMES_IN_FLIGHT}}, 1 * gpu::MAX_FRAMES_IN_FLIGHT);
}

void ParticleRenderPass::destroyFrameResources()
{
    vk::Device device = m_core->getDevice();
    for (auto framebuffer : framebuffers)
    {
        device.destroyFramebuffer(framebuffer);
    }
    device.freeCommandBuffers(m_core->getCommandPool(), commandBuffers);

    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++)
    {
        m_core->destroyBuffer(uniformBuffers[i]);
    }

    m_core->destroyDescriptorPool(descriptorPool);
}
void ParticleRenderPass::destroy()
{
    destroyFrameResources();

    vk::Device device = m_core->getDevice();

    device.destroyShaderModule(fragShaderModule);
    device.destroyShaderModule(vertShaderModule);
    device.destroyShaderModule(geomShaderModule);

    particleModel.destroy();
    m_core->destroyBuffer(particleModelIndexBuffer);
    m_core->destroyBuffer(particleModelVertexBuffer);
        
    device.destroyPipelineLayout(pipelineLayout);
    device.destroyPipeline(graphicsPipeline);
    device.destroyRenderPass(renderPass);

    m_core->destroyDescriptorSetLayout(descriptorSetLayout);
}
void ParticleRenderPass::updateUniformBuffer(uint32_t currentImage)
{

    UniformBufferObject ubo{};
    ubo.model = glm::mat4(1.0f);
    ubo.view = m_camera->getView();
    ubo.proj = glm::perspective(glm::radians(45.0f), m_core->getSwapChainExtent().width / (float)m_core->getSwapChainExtent().height, 0.1f, 1000.0f);
    ubo.proj[1][1] *= -1;

    m_core->updateBufferData(uniformBuffers[currentImage], &ubo, (size_t) sizeof(ubo));
}
