#define _USE_MATH_DEFINES
#include <math.h>
#include "core.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include "compute_pass.h"

struct Particle{
  glm::vec2 position = glm::vec2(0,0);
  glm::vec2 boundaryNormal = glm::vec2(0,0);
  glm::vec2 velocity = glm::vec2(0,0);
  glm::vec2 force = glm::vec2(0,0);
  float rho = 0.0;
  float p = 0.0;
  float deltaB = 0.0;
  float pad0 = 0.0;
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


struct SPHSettings{
  glm::vec2 G = glm::vec2(0.f, -10.f);                 // external (gravitational) forces
  float PI = (float)M_PI;
  float rhoRest = 300.f;  // rest density
  float GAS_CONST = 2000.f; // const for equation of state
  float kernelRadius = 16.f;		   // kernel radius
  float kernelRadiusSquared = kernelRadius * kernelRadius;		   // radius^2 for optimization
  float MASS = 2.5f;		   // assume all particles have the same mass
  float VISC = 200.f;	   // viscosity constant
  float dt = 0.0007f;	   // integration timestep

  // smoothing kernels defined in Müller and their gradients
  // adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.
  float POLY6 = 4.f / ((float)M_PI * pow(kernelRadius, 8.f));
  float SPIKY_GRAD = -10.f / ((float)M_PI * pow(kernelRadius, 5.f));
  float VISC_LAP = 40.f / ((float)M_PI * pow(kernelRadius, 5.f));

  // simulation parameters
  float BOUNDARY_EPSILON = kernelRadius; // boundary epsilon
  float BOUNDARY_DAMPING = -0.1f;
  float DOMAIN_WIDTH = 1.5 * 800.f;
  float DOMAIN_HEIGHT = 1.5 * 600.f;

  float pad0, pad1, pad2;
};

class GranularMatter
{
private:
    gpu::Core* m_core;

    std::vector<vk::CommandBuffer> commandBuffers;
    
    std::vector<vk::Buffer> particlesBufferA;
    std::vector<vk::Buffer> settingsBuffer;
    glm::ivec3 computeSpace = glm::ivec3(64, 64, 1);

    vk::DescriptorSetLayout descriptorSetLayout;
    std::vector<vk::DescriptorSet> descriptorSets;
    vk::DescriptorPool descriptorPool;
 
    gpu::ComputePass boundaryUpdatePass;
    gpu::ComputePass densityPressurePass;
    gpu::ComputePass forcePass;
    gpu::ComputePass integratePass;

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
    void update(int currentFrame, int imageIndex);
    void destroyFrameResources();
    void destroy(); 
};

