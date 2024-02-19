#include "particle_renderpass.h"
#include <chrono>
#include "global.h"
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/gtc/matrix_transform.hpp>
#include "initializers.h"
#include "granular_matter.h"

using namespace gpu;

ParticleRenderPass::ParticleRenderPass(gpu::Core *core, gpu::Camera *camera)
{
    _core = core;
    m_camera = camera;

    particleModel = Model();
    particleModel.load_from_glb(ASSETS_PATH "/models/grain_smooth.glb");

    std::vector<unsigned int> remap(particleModel._indices.size()); // allocate temporary memory for the remap table
    const size_t max_vertices = 64;
    const size_t max_triangles = 124;
    const float cone_weight = 0.0f;

    size_t max_meshlets = meshopt_buildMeshletsBound(particleModel._indices.size(), max_vertices, max_triangles);
    _meshlets = std::vector<meshopt_Meshlet>(max_meshlets);
    _meshletVertices = std::vector<unsigned int> (max_meshlets * max_vertices);
    _meshletTriangles = std::vector<unsigned char> (max_meshlets * max_triangles * 3);

    size_t meshlet_count = meshopt_buildMeshlets(_meshlets.data(), _meshletVertices.data(), _meshletTriangles.data(),  particleModel._indices.data(),  particleModel._indices.size(), glm::value_ptr(particleModel._vertices.data()[0].pos), particleModel._vertices.size(), sizeof(Vertex), max_vertices, max_triangles, cone_weight);

    const meshopt_Meshlet& last = _meshlets[meshlet_count - 1];

    _meshletVertices.resize(last.vertex_offset + last.vertex_count);
    _meshletTriangles.resize(last.triangle_offset + ((last.triangle_count * 3 + 3) & ~3));
    _meshlets.resize(meshlet_count);
    _meshletBounds.resize(meshlet_count);

    // generate bounds for meshlets
    for(int i = 0; i < meshlet_count; i++){
        auto& m = _meshlets[i];
        _meshletBounds[i] = meshopt_computeMeshletBounds(&_meshletVertices[m.vertex_offset], &_meshletTriangles[m.triangle_offset],
        m.triangle_count, glm::value_ptr(particleModel._vertices.data()[0].pos), particleModel._vertices.size(), sizeof(Vertex));  
    }
    std::cout << "[model] " << "vertices: " << particleModel._vertices.size() << " indices: " << particleModel._indices.size() << " meshlets: " << meshlet_count << std::endl;

    // particleModelIndexBuffer = _core->bufferFromData(particleModel._indices.data(), particleModel._indices.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eIndexBuffer, vma::MemoryUsage::eAutoPreferDevice);
    particleModelVertexBuffer = _core->bufferFromData(particleModel._vertices.data(), particleModel._vertices.size() * sizeof(Vertex), vk::BufferUsageFlagBits::eStorageBuffer, vma::MemoryUsage::eAutoPreferDevice);
    _meshletBuffer = _core->bufferFromData(_meshlets.data(), _meshlets.size() * sizeof(meshopt_Meshlet), vk::BufferUsageFlagBits::eStorageBuffer, vma::MemoryUsage::eAutoPreferDevice);
    _meshletBoundsBuffer = _core->bufferFromData(_meshletBounds.data(), _meshletBounds.size() * sizeof(meshopt_Bounds), vk::BufferUsageFlagBits::eStorageBuffer, vma::MemoryUsage::eAutoPreferDevice);
    _meshletVerticesBuffer = _core->bufferFromData(_meshletVertices.data(), _meshletVertices.size() * sizeof(unsigned int), vk::BufferUsageFlagBits::eStorageBuffer, vma::MemoryUsage::eAutoPreferDevice);
    _meshletTrianglesBuffer = _core->bufferFromData(_meshletTriangles.data(), _meshletTriangles.size() * sizeof(unsigned char), vk::BufferUsageFlagBits::eStorageBuffer, vma::MemoryUsage::eAutoPreferDevice);
}
void ParticleRenderPass::init()
{

    meshShaderModule = _core->loadShaderModule(SHADER_PATH "/shader.mesh");
    taskShaderModule = _core->loadShaderModule(SHADER_PATH "/shader.task");
    fragShaderModule = _core->loadShaderModule(SHADER_PATH "/shader.frag");
    geomShaderModule = _core->loadShaderModule(SHADER_PATH"/shader.geom");

    descriptorSetLayout = _core->createDescriptorSetLayout({
        {0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eFragment},
        {1, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eFragment},
        {2, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eFragment},
        {3, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eFragment},
        {4, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eFragment},
        {5, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eFragment},
        {6, vk::DescriptorType::eStorageBuffer, vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT | vk::ShaderStageFlagBits::eFragment},
    });
    
    _renderContext = RenderContext(_core, vk::AttachmentLoadOp::eClear, vk::AttachmentStoreOp::eStore);
    createGraphicsPipeline();


    initFrameResources();
}
void ParticleRenderPass::initFrameResources()
{
    _renderContext.initFramebuffers();

    uniformBuffers.resize(gpu::MAX_FRAMES_IN_FLIGHT);
    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++)
    {
        uniformBuffers[i] = _core->createBuffer(sizeof(UniformBufferObject), vk::BufferUsageFlagBits::eUniformBuffer | vk::BufferUsageFlagBits::eTransferDst, vma::MemoryUsage::eAutoPreferDevice, vma::AllocationCreateFlagBits::eHostAccessSequentialWrite);
    }

    createDescriptorPool();
    createDescriptorSets();
    _renderContext.initCommandBuffers();
}

void ParticleRenderPass::update(int imageIndex, float dt)
{
    UniformBufferObject ubo{};
    ubo.model = glm::mat4(1.0f);
    ubo.view = m_camera->getView();
    ubo.proj = glm::perspective(glm::radians(45.0f), _core->getSwapchainExtent().width / (float)_core->getSwapchainExtent().height, 0.1f, 1000.0f);
    ubo.proj[1][1] *= -1;

    _core->updateBufferData(uniformBuffers[_core->_swapchainContext._currentFrame], &ubo, (size_t) sizeof(ubo));

    _renderContext.beginCommandBuffer();
    _renderContext.beginRenderPass(imageIndex);

    _renderContext.getCommandBuffer().bindPipeline(vk::PipelineBindPoint::eGraphics, graphicsPipeline);

    _renderContext.getCommandBuffer().bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[_core->_swapchainContext._currentFrame], 0, nullptr);

    _renderContext.getCommandBuffer().pushConstants(pipelineLayout, vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT, 0, sizeof(SPHSettings), &settings);


    size_t numTasks = 1; //static_cast<size_t>(std::ceil(_meshlets.size() / 32.f));
    size_t numInstances = static_cast<size_t>(std::ceil(vertexCount / 32.f));
    _renderContext.getCommandBuffer().drawMeshTasksEXT( numInstances, numTasks,1);


    _renderContext.endRenderPass();
    _renderContext.endCommandBuffer();

}

void ParticleRenderPass::createDescriptorSets()
{

    descriptorSets = _core->allocateDescriptorSets(descriptorSetLayout, descriptorPool, gpu::MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++)
    {
        _core->addDescriptorWrite(descriptorSets[i], {0, vk::DescriptorType::eUniformBuffer, uniformBuffers[i], sizeof(UniformBufferObject)});
        _core->addDescriptorWrite(descriptorSets[i], {1, vk::DescriptorType::eStorageBuffer, particleModelVertexBuffer, particleModel._vertices.size() * sizeof(Vertex)});
        _core->addDescriptorWrite(descriptorSets[i], {2, vk::DescriptorType::eStorageBuffer, _meshletBuffer, _meshlets.size() * sizeof(meshopt_Meshlet)});
        _core->addDescriptorWrite(descriptorSets[i], {3, vk::DescriptorType::eStorageBuffer, _meshletVerticesBuffer, _meshletVertices.size() * sizeof(unsigned int)});
        _core->addDescriptorWrite(descriptorSets[i], {4, vk::DescriptorType::eStorageBuffer, _meshletTrianglesBuffer, _meshletTriangles.size() * sizeof(unsigned char)});
        _core->addDescriptorWrite(descriptorSets[i], {5, vk::DescriptorType::eStorageBuffer, _meshletBoundsBuffer, _meshletBounds.size() * sizeof(meshopt_Bounds)});
        _core->addDescriptorWrite(descriptorSets[i], {6, vk::DescriptorType::eStorageBuffer, vertexBuffer, vertexCount * sizeof(HRParticle)});
        _core->updateDescriptorSet(descriptorSets[i]);
    }
}

void ParticleRenderPass::createGraphicsPipeline()
{
    vk::PipelineShaderStageCreateInfo taskShaderStageInfo({}, vk::ShaderStageFlagBits::eTaskEXT, taskShaderModule, "main");
    vk::PipelineShaderStageCreateInfo meshShaderStageInfo({}, vk::ShaderStageFlagBits::eMeshEXT, meshShaderModule, "main");
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo({}, vk::ShaderStageFlagBits::eFragment, fragShaderModule, "main");

    std::vector<vk::PipelineShaderStageCreateInfo> shaderStages = {
        taskShaderStageInfo, 
        meshShaderStageInfo, 
        fragShaderStageInfo
    }; 

    std::vector<vk::VertexInputBindingDescription> bindings;
    std::vector<vk::VertexInputAttributeDescription> attributes;

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo({}, bindings, attributes);
    vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, VK_FALSE);
    vk::PipelineRasterizationStateCreateInfo rasterizer({}, VK_FALSE, VK_FALSE, vk::PolygonMode::eFill, vk::CullModeFlagBits::eBack, vk::FrontFace::eCounterClockwise, VK_FALSE, 0.0f, 0.0f, 0.0f, 1.0f);
    vk::PushConstantRange pushConstantRange{vk::ShaderStageFlagBits::eFragment | vk::ShaderStageFlagBits::eMeshEXT | vk::ShaderStageFlagBits::eTaskEXT, 0, sizeof(SPHSettings)};
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
    _core->getDevice().destroyShaderModule(taskShaderModule);
    _core->getDevice().destroyShaderModule(meshShaderModule);
    _core->getDevice().destroyShaderModule(geomShaderModule);

    particleModel.destroy();
    // _core->destroyBuffer(particleModelIndexBuffer);
    _core->destroyBuffer(particleModelVertexBuffer);

    _core->destroyBuffer(_meshletBuffer);
    _core->destroyBuffer(_meshletBoundsBuffer);
    _core->destroyBuffer(_meshletVerticesBuffer);
    _core->destroyBuffer(_meshletTrianglesBuffer);
        
    _core->getDevice().destroyPipelineLayout(pipelineLayout);
    _core->getDevice().destroyPipeline(graphicsPipeline);
    _renderContext.destroyRenderPass();

    _core->destroyDescriptorSetLayout(descriptorSetLayout);
}
