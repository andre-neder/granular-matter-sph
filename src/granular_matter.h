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
  static std::array<vk::VertexInputAttributeDescription, 1> getAttributeDescriptions() {
      std::array<vk::VertexInputAttributeDescription, 1> attributeDescriptions{
          vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, position)),
          // vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Particle, color)),
          // vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Particle, texCoord))
      };
      return attributeDescriptions;
  }
};
  //? sphere Volume in dm^3 * density
  //   float MASS = (4.f / 3.f * (float) M_PI * (particleRadius * particleRadius * particleRadius)) * rhoRest; 
  //? sphere Volume in 2D 
  // float mass = volume * rhoRest; //* kg
#define particleRadius 0.001f //* m   //? 0.063 bis 2 mm // 1dm = 10cm = 100mm
#define volume (float) M_PI * (particleRadius * particleRadius)

struct SPHSettings{
    glm::vec2 G = glm::vec2(0.f, -9.81f); //* m/s^2
    float rhoRest = 1.5f; //* kg / m^3 -> 2D kg / m^2  //? 1.5 - 2.2 kg / dm^3 (1950kg/m^3)
    float kernelRadius = 0.3f; // *m	

    float mass = 1.f;
    float stiffness = 25.f;	  
    float dt = 0.000f;	  
    float DOMAIN_WIDTH = 9.f; //* m

    float DOMAIN_HEIGHT = 9.f; //* m
    float pad0, pad1, pad2;
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

    SPHSettings settings;

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

