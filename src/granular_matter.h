#define _USE_MATH_DEFINES
#include <math.h>
#include "core.h"
#include <glm/glm.hpp>
#include "compute_pass.h"
#include "global.h"
#include "rigidbody.h"

struct BitonicSortParameters {
    enum eAlgorithmVariant : uint32_t {
        eLocalBitonicMergeSortExample = 0,
        eLocalDisperse                = 1,
        eBigFlip                      = 2,
        eBigDisperse                  = 3,
    };
    uint32_t          h = 0;
    eAlgorithmVariant algorithm;
};

struct ParticleGridEntry{
    uint32_t particleIndex = 0;
    uint32_t cellKey = UINT32_MAX;
};

struct LRParticle{
    glm::vec4 position = glm::vec4(0);
    glm::vec4 velocity = glm::vec4(0);
    glm::vec4 externalForce = glm::vec4(0);
    glm::vec4 internalForce = glm::vec4(0);
    glm::vec4 d = glm::vec4(0);
    glm::vec4 dijpj = glm::vec4(0);
    glm::mat4 stress = glm::mat4(0.0);
    glm::mat4 deviatoricStress = glm::mat4(0.0);

    float rho = settings.rho0;
    float p = 0.0;
    float V = 0.0;
    float a = 0.0;

    float dpi = 0.0;
    float lastP = 0.0;
    float densityAdv = 0.0;
    float pad0 = 0.0;

    glm::vec4 averageN;
    glm::vec4 color;

    LRParticle(){};
    inline LRParticle(float x, float y , float z) { position = glm::vec4(x, y, z, 1.0); }
    static const uint32_t BINDING = 1;
    static std::array<vk::VertexInputBindingDescription, 1> getBindingDescription() {
        std::array<vk::VertexInputBindingDescription, 1> bindingDescriptions = {
            vk::VertexInputBindingDescription(BINDING, sizeof(LRParticle), vk::VertexInputRate::eInstance)
        };
        return bindingDescriptions;
    }
    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{
            vk::VertexInputAttributeDescription(7, BINDING, vk::Format::eR32G32B32Sfloat, offsetof(LRParticle, position)),
            vk::VertexInputAttributeDescription(8, BINDING, vk::Format::eR32G32B32Sfloat, offsetof(LRParticle, averageN)),
            vk::VertexInputAttributeDescription(9, BINDING, vk::Format::eR32G32B32A32Sfloat, offsetof(LRParticle, color)),
        };
        return attributeDescriptions;
    }
};

struct HRParticle{
    glm::vec4 position = glm::vec4(0);
    glm::vec4 velocity = glm::vec4(0);  
    glm::vec4 color = glm::vec4(0);  
    HRParticle(){};
    inline HRParticle(float x, float y, float z) { position = glm::vec4(x, y, z, 1); }
    static const uint32_t BINDING = 1;
     static std::array<vk::VertexInputBindingDescription, 1> getBindingDescription() {
        std::array<vk::VertexInputBindingDescription, 1> bindingDescriptions = {
            vk::VertexInputBindingDescription(BINDING, sizeof(HRParticle), vk::VertexInputRate::eInstance)
        };
        return bindingDescriptions;
    }
    static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 3> attributeDescriptions{
            vk::VertexInputAttributeDescription(7, BINDING, vk::Format::eR32G32B32A32Sfloat, offsetof(HRParticle, position)),
            vk::VertexInputAttributeDescription(8, BINDING, vk::Format::eR32G32B32A32Sfloat, offsetof(HRParticle, velocity)),
            vk::VertexInputAttributeDescription(9, BINDING, vk::Format::eR32G32B32A32Sfloat, offsetof(HRParticle, color)),
        };
        return attributeDescriptions;
    }
};

struct VolumeMapTransform{
    glm::vec4 position = glm::vec4(0.0);
    glm::vec4 scale = glm::vec4(1.0);

    inline void disable(){ position.w = 0.0; };
    inline void enable(){ position.w = 1.0; };
};

struct AdditionalData{
    glm::mat4 D = glm::mat4(0.0);
    float averageDensityError = 0.f;
    uint32_t frameIndex = 0;
    float pad[2];
};

class GranularMatter
{
public:
    GranularMatter(/* args */){};
    GranularMatter(gpu::Core* core);
    ~GranularMatter();

    std::vector<LRParticle> lrParticles;
    std::vector<HRParticle> hrParticles;
    vk::Buffer particlesBufferB;
    vk::Buffer particlesBufferHR;

    inline vk::CommandBuffer getCommandBuffer(int index){ return commandBuffers[index]; };
    void initFrameResources();
    void update(int currentFrame, int imageIndex, float dt);
    void destroyFrameResources();
    void destroy(); 
    void init();

    void createSignedDistanceFields();

    std::vector<vk::Semaphore> iisphSemaphores;

    std::vector<RigidBody2D*> rigidBodies;
    std::vector<VolumeMapTransform> volumeMapTransforms;
    void createDescriptorSets();
    void updateVolumeMapTransforms();
private:
    gpu::Core* m_core;
    

    AdditionalData additionalData;
    std::vector<vk::Buffer>additionalDataBuffer;
    std::vector<vk::Fence> iisphFences;


    std::vector<vk::CommandBuffer> commandBuffers;
    
    vk::Buffer volumeMapTransformsBuffer;
    
    std::vector<ParticleGridEntry> particleCells; // particle (index) is in cell (value)
    std::vector<uint32_t> startingIndices; 
    
    vk::Buffer particleCellBuffer;
    vk::Buffer startingIndicesBuffers;

    std::vector<vk::DescriptorSet> descriptorSetsGrid;
    std::vector<vk::DescriptorSet> descriptorSetsParticles;

    vk::DescriptorSetLayout descriptorSetLayoutGrid;
    vk::DescriptorSetLayout descriptorSetLayoutParticles;

    vk::DescriptorPool descriptorPool;
 
    gpu::ComputePass initPass;
    gpu::ComputePass bitonicSortPass;
    gpu::ComputePass startingIndicesPass;
    gpu::ComputePass computeDensityPass;
    gpu::ComputePass computeSurfaceNormalPass;

    gpu::ComputePass iisphvAdvPass;
    gpu::ComputePass iisphRhoAdvPass;
    gpu::ComputePass iisphdijpjSolvePass;
    gpu::ComputePass iisphPressureSolvePass;
    gpu::ComputePass iisphSolveEndPass;

    gpu::ComputePass computeStressPass;
    gpu::ComputePass computeInternalForcePass;
    gpu::ComputePass integratePass;
    gpu::ComputePass advectionPass;

    std::vector<vk::Image> signedDistanceFields;
    std::vector<vk::ImageView> signedDistanceFieldViews;
    vk::Sampler volumeMapSampler;

    void createCommandBuffers();
    void createDescriptorSetLayout();
    void createDescriptorPool();
    

};

