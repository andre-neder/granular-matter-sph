#define _USE_MATH_DEFINES
#include <math.h>
#include "core.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "compute_pass.h"
#include "global.h"


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
    glm::vec2 position = glm::vec2(0,0);
    glm::vec2 predPosition = glm::vec2(0,0);
    // 4
    glm::vec2 velocity = glm::vec2(0,0);
    glm::vec2 externalForce = glm::vec2(0,0);
    // 8
    glm::vec2 internalForce = glm::vec2(0,0);
    float rho = settings.rho0;
    float p = 0.0;
    // 12
    float V = 0.0;
    float a = 0.0;
    glm::vec2 d = glm::vec2(0,0);
    // 16
    glm::mat2 stress = glm::mat2(1.0);
    // 20
    glm::vec2 dijpj = glm::vec2(0,0);
    float dpi = 0.0;
    float lastP = 0.0;
    //24
    float densityAdv = 0.0;
    float pad0;
    float pad1;
    float pad2;

    LRParticle(){};
    inline LRParticle(float x, float y) { position = glm::vec2(x, y); }

    static std::array<vk::VertexInputBindingDescription, 1> getBindingDescription() {
        std::array<vk::VertexInputBindingDescription, 1> bindingDescriptions = {
            vk::VertexInputBindingDescription(0, sizeof(LRParticle), vk::VertexInputRate::eVertex)
        };
        return bindingDescriptions;
    }
    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(LRParticle, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(LRParticle, velocity)),
            // vk::VertexInputAttributeDescription(1, 0, vk::Format::eR16Sfloat, offsetof(LRParticle, rho)),
            // vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(LRParticle, texCoord))
        };
        return attributeDescriptions;
    }
};

struct HRParticle{
    glm::vec2 position = glm::vec2(0,0);
    glm::vec2 velocity = glm::vec2(0,0);  
    HRParticle(){};
    inline HRParticle(float x, float y) { position = glm::vec2(x, y); }

     static std::array<vk::VertexInputBindingDescription, 1> getBindingDescription() {
        std::array<vk::VertexInputBindingDescription, 1> bindingDescriptions = {
            vk::VertexInputBindingDescription(0, sizeof(HRParticle), vk::VertexInputRate::eVertex)
        };
        return bindingDescriptions;
    }
    static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
        std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{
            vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(HRParticle, position)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32Sfloat, offsetof(HRParticle, velocity)),
            // vk::VertexInputAttributeDescription(1, 0, vk::Format::eR16Sfloat, offsetof(LRParticle, rho)),
            // vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(LRParticle, texCoord))
        };
        return attributeDescriptions;
    }
};

namespace std{
    template <typename T> int sign(T val) {
        return (T(0) < val) - (val < T(0));
    }
}

struct AABB{
    glm::vec2 min;
    glm::vec2 max;
};

struct RigidBody2D{
    bool active = false; // states if object is influenced by forces
    bool invert = false; // states uf the sdf should be inverted
    glm::vec2 position = glm::vec2(0.0);
    glm::vec2 scale = glm::vec2(1.0);
    AABB aabb;
    virtual glm::vec2 signedDistanceGradient(glm::vec2 position) = 0; // calculates the signed distance and direction 
    virtual float signedDistance(glm::vec2 position) = 0; // calculates the signed distance 
};

struct Box2D : public RigidBody2D{
    glm::vec2 halfSize;
    inline Box2D(glm::vec2 halfSize) : halfSize(halfSize) {
        aabb.min = -halfSize;
        aabb.max = halfSize;
    };
    glm::vec2 signedDistanceGradient(glm::vec2 p) override {
        glm::vec2 q = glm::abs(p) - halfSize;
        glm::vec2 n = (q.x > q.y ?  glm::vec2(std::sign(p.x), 0) : glm::vec2(0, std::sign(p.y)));
        return n * glm::length(glm::max(q,glm::vec2(0.0))) + std::min(std::max(q.x, q.y), 0.f);
    };
    float signedDistance(glm::vec2 p) override {
        glm::vec2 q = glm::abs(p) - halfSize;
        return glm::length(glm::max(q,glm::vec2(0.0))) + std::min(std::max(q.x, q.y), 0.f);
    };
};

struct Line2D : public RigidBody2D{
    glm::vec2 normal;
    float h; 
    inline Line2D(glm::vec2 normal, float h) : normal(normal), h(h) {
        aabb.min = glm::vec2(0);
        aabb.max = glm::vec2(0);
    };
    glm::vec2 signedDistanceGradient(glm::vec2 p) override {
        return normal * (glm::dot(p, normal) + h);
    };
    float signedDistance(glm::vec2 p) override {
        return (glm::dot(p, normal) + h);
    };
};

struct VolumeMapTransform{
    glm::vec2 position = glm::vec2(0.0);
    glm::vec2 scale = glm::vec2(1.0);
};

struct AdditionalData{
    float averageDensityError = 0.01f;
    float pad[3];
};

class GranularMatter
{
public:
    GranularMatter(/* args */){};
    GranularMatter(gpu::Core* core);
    ~GranularMatter();

    std::vector<LRParticle> lrParticles;
    std::vector<HRParticle> hrParticles;
    std::vector<vk::Buffer> particlesBufferB;
    std::vector<vk::Buffer> particlesBufferHR;

    inline vk::CommandBuffer getCommandBuffer(int index){ return commandBuffers[index]; };
    void initFrameResources();
    void update(int currentFrame, int imageIndex);
    void destroyFrameResources();
    void destroy(); 

private:
    gpu::Core* m_core;
    

    AdditionalData additionalData;
    std::vector<vk::Buffer>additionalDataBuffer;

    std::vector<VolumeMapTransform> volumeMapTransforms;
    std::vector<vk::CommandBuffer> commandBuffers;
    
    std::vector<vk::Buffer> particlesBufferA;
    std::vector<vk::Buffer> volumeMapTransformsBuffer;
    
    std::vector<ParticleGridEntry> particleCells; // particle (index) is in cell (value)
    std::vector<vk::Buffer> particleCellBuffer;
    std::vector<uint32_t> startingIndices; 
    std::vector<vk::Buffer> startingIndicesBuffers;

    std::vector<vk::DescriptorSet> descriptorSetsGrid;
    std::vector<vk::DescriptorSet> descriptorSetsParticles;

    vk::DescriptorSetLayout descriptorSetLayoutGrid;
    vk::DescriptorSetLayout descriptorSetLayoutParticles;

    vk::DescriptorPool descriptorPool;
 
    gpu::ComputePass initPass;
    gpu::ComputePass bitonicSortPass;
    gpu::ComputePass startingIndicesPass;
    gpu::ComputePass computeDensityPass;

    gpu::ComputePass iisphvAdvPass;
    gpu::ComputePass iisphRhoAdvPass;
    gpu::ComputePass iisphdijpjSolvePass;
    gpu::ComputePass iisphPressureSolvePass;
    gpu::ComputePass iisphSolveEndPass;

    gpu::ComputePass computeStressPass;
    gpu::ComputePass computeInternalForcePass;
    gpu::ComputePass integratePass;
    gpu::ComputePass advectionPass;

    std::vector<RigidBody2D*> rigidBodies;
    std::vector<vk::Image> signedDistanceFields;
    std::vector<vk::ImageView> signedDistanceFieldViews;
    vk::Sampler volumeMapSampler;

    void createCommandBuffers();
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();
    void createSignedDistanceFields();

};

