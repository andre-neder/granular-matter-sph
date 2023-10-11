#define _USE_MATH_DEFINES
#include <math.h>
#include "core.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "compute_pass.h"

struct Particle{
  glm::vec2 position = glm::vec2(0,0);
  glm::vec2 predPosition = glm::vec2(0,0);
  // 4
  glm::vec2 velocity = glm::vec2(0,0);
  glm::vec2 predVelocity = glm::vec2(0,0);
  // 8
  glm::vec2 pressureAcceleration = glm::vec2(0,0);
  float rho = 0.0;
  float p = 0.0;
  // 12
  float V = 0.0;
  float pad0 = 0.0;
  float pad1 = 0.0;
  float pad2 = 0.0;
  // 16
  Particle(){};
  inline Particle(float x, float y) { position = glm::vec2(x, y); }

  static std::array<vk::VertexInputBindingDescription, 1> getBindingDescription() {
      std::array<vk::VertexInputBindingDescription, 1> bindingDescriptions = {
          vk::VertexInputBindingDescription(0, sizeof(Particle), vk::VertexInputRate::eVertex)
      };
      return bindingDescriptions;
  }
  static std::array<vk::VertexInputAttributeDescription, 2> getAttributeDescriptions() {
      std::array<vk::VertexInputAttributeDescription, 2> attributeDescriptions{
          vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position)),
          vk::VertexInputAttributeDescription(1, 0, vk::Format::eR16Sfloat, offsetof(Particle, rho)),
          // vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, texCoord))
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
    glm::ivec3 computeSpace = glm::ivec3(32, 32, 1);

    vk::DescriptorSetLayout descriptorSetLayout;
    std::vector<vk::DescriptorSet> descriptorSets;
    vk::DescriptorPool descriptorPool;
 
    gpu::ComputePass boundaryUpdatePass;
    gpu::ComputePass initPass;
    gpu::ComputePass predictPositionPass;
    gpu::ComputePass predictDensityPass;
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

    std::vector<Particle> particles;
    std::vector<Particle> boundaryParticles;
    std::vector<vk::Buffer> particlesBufferB;
    std::vector<vk::Buffer> boundaryParticlesBuffer;

    inline vk::CommandBuffer getCommandBuffer(int index){ return commandBuffers[index]; };
    void initFrameResources();
    void updateSettings(float dt, int currentFrame);
    void update(int currentFrame, int imageIndex);
    void destroyFrameResources();
    void destroy(); 
};

