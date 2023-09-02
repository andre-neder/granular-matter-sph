#define _USE_MATH_DEFINES
#include <math.h>
#include "core.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

struct Particle{
  glm::vec2 position = glm::vec2(0,0);
  glm::vec2 velocity = glm::vec2(0,0);
  glm::vec2 force = glm::vec2(0,0);
  float rho = 0.0;
  float p = 0.0;
  Particle(){};
  inline Particle(float x, float y) { position = glm::vec2(x, y); }
};


struct SPHSettings{
  glm::vec2 G = glm::vec2(0.f, -10.f);                 // external (gravitational) forces
  float PI = (float)M_PI;
  float REST_DENS = 300.f;  // rest density
  float GAS_CONST = 2000.f; // const for equation of state
  float KERNEL_RADIUS = 16.f;		   // kernel radius
  float KERNEL_RADIUS_SQ = KERNEL_RADIUS * KERNEL_RADIUS;		   // radius^2 for optimization
  float MASS = 2.5f;		   // assume all particles have the same mass
  float VISC = 200.f;	   // viscosity constant
  float DT = 0.0007f;	   // integration timestep

  // smoothing kernels defined in Müller and their gradients
  // adapted to 2D per "SPH Based Shallow Water Simulation" by Solenthaler et al.
  float POLY6 = 4.f / ((float)M_PI * pow(KERNEL_RADIUS, 8.f));
  float SPIKY_GRAD = -10.f / ((float)M_PI * pow(KERNEL_RADIUS, 5.f));
  float VISC_LAP = 40.f / ((float)M_PI * pow(KERNEL_RADIUS, 5.f));

  // simulation parameters
  float BOUNDARY_EPSILON = KERNEL_RADIUS; // boundary epsilon
  float BOUNDARY_DAMPING = -0.5f;
  float DOMAIN_WIDTH = 1.5 * 800.f;
  float DOMAIN_HEIGHT = 1.5 * 600.f;

  float pad0, pad1, pad2;
};

class GranularMatter
{
private:
    gpu::Core* m_core;

    std::vector<vk::CommandBuffer> commandBuffers;
    
    std::vector<Particle> particles;
    std::vector<vk::Buffer> particlesBufferA;
    std::vector<vk::Buffer> particlesBufferB;
    std::vector<vk::Buffer> settingsBuffer;

    vk::DescriptorSetLayout descriptorSetLayout;
    std::vector<vk::DescriptorSet> descriptorSets;
    vk::DescriptorPool descriptorPool;

    vk::ShaderModule densityPressureModule;
    vk::ShaderModule forceModule;
    vk::ShaderModule integrateModule;

    vk::PipelineLayout densityPressureLayout;
    vk::PipelineLayout forceLayout;
    vk::PipelineLayout integrateLayout;

    vk::Pipeline densityPressurePipeline;
    vk::Pipeline forcePipeline;
    vk::Pipeline integratePipeline;

    SPHSettings settings;

    void createCommandBuffers();
    void createDescriptorSetLayout();
    void createDescriptorPool();
    void createDescriptorSets();
    void createComputePipeline();
public:
    GranularMatter(/* args */){};
    GranularMatter(gpu::Core* core);
    ~GranularMatter();

    inline vk::CommandBuffer getCommandBuffer(int index){ return commandBuffers[index]; };
    void initFrameResources();
    void update(int imageIndex);
    void destroyFrameResources();
    void destroy(); 
};
