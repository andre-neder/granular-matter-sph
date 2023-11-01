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

struct HRParticle{
    glm::vec2 position = glm::vec2(0,0);
    glm::vec2 predPosition = glm::vec2(0,0);
    // 4
    glm::vec2 velocity = glm::vec2(0,0);
    glm::vec2 predVelocity = glm::vec2(0,0);
    // 8
    glm::vec2 internalForce = glm::vec2(0,0);
    float rho = settings.rho0;
    float p = 0.0;
    // 12
    float V = 0.0;
    float boundaryVolume = 0.0;
    glm::vec2 boundaryNormal = glm::vec2(0,0);
    // 16
    glm::mat2 stress = glm::mat2(1.0);
    // 20
    uint32_t fluidNeighbors[31];
    uint32_t fluidNeighborCount;
    //52

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
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32Sfloat, offsetof(HRParticle, rho)),
            //   vk::VertexInputAttributeDescription(1, 0, vk::Format::eR16Sfloat, offsetof(HRParticle, rho)),
            // vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(HRParticle, texCoord))
        };
        return attributeDescriptions;
    }
};

class GranularMatter
{
private:
    gpu::Core* m_core;
    

    std::vector<vk::CommandBuffer> commandBuffers;
    
    std::vector<vk::Buffer> particlesBufferA;
    std::vector<vk::Buffer> settingsBuffer;
    
    std::vector<ParticleGridEntry> particleCells; // particle (index) is in cell (value)
    std::vector<vk::Buffer> particleCellBuffer;
    std::vector<uint32_t> startingIndices; 
    std::vector<vk::Buffer> startingIndicesBuffers;
    // vk::DescriptorSetLayout descriptorSetLayoutCell;
    std::vector<vk::DescriptorSet> descriptorSetsGrid;
    // vk::DescriptorPool descriptorPoolCell;

    // vk::DescriptorSetLayout descriptorSetLayout;
    std::vector<vk::DescriptorSet> descriptorSetsParticles;
    // vk::DescriptorPool descriptorPool;

    vk::DescriptorSetLayout descriptorSetLayoutGrid;
    vk::DescriptorSetLayout descriptorSetLayoutParticles;

    vk::DescriptorPool descriptorPool;
 
    gpu::ComputePass initPass;
    gpu::ComputePass bitonicSortPass;
    gpu::ComputePass startingIndicesPass;
    gpu::ComputePass predictDensityPass;
    gpu::ComputePass predictStressPass;
    gpu::ComputePass predictForcePass;
    gpu::ComputePass applyPass;

    void createCommandBuffers();
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();
public:
    GranularMatter(/* args */){};
    GranularMatter(gpu::Core* core);
    ~GranularMatter();

    std::vector<HRParticle> particles;
    std::vector<vk::Buffer> particlesBufferB;

    inline vk::CommandBuffer getCommandBuffer(int index){ return commandBuffers[index]; };
    void initFrameResources();
    void update(int currentFrame, int imageIndex);
    void destroyFrameResources();
    void destroy(); 
};

