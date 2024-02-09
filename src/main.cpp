#pragma once

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#define GLM_ENABLE_EXPERIMENTAL

#define TINYGLTF_USE_CPP14

#ifdef _WIN32
    #include <windows.h>
#endif

#include <stb_image.h>
#include <stb_image_write.h>
#include <tiny_gltf.h>
#include <functional>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>

#include "particle_renderpass.h"
#include "imgui_renderpass.h"
#include "triangle_renderpass.h"
#include "granular_matter.h"

#include "global.h"
#include "camera.h"
#include "input.h"

const uint32_t WIDTH = 1280;
const uint32_t HEIGHT = 720;

class Application {
public:
    void run() {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }
private:
    vk::Device device;

    gpu::Core core;
    gpu::Camera camera;
    gpu::Window window;
    gpu::InputManager input;

    gpu::ComputeBundle computeBundle;

    gpu::ParticleRenderPass particleRenderPass;
    gpu::TriangleRenderPass triangleRenderPass;
    gpu::ImguiRenderPass imguiRenderPass;

    GranularMatter simulation;

    Model dumpTruckModel;
    Model planeModel;
    Model hourglasModel;

    Mesh3D dumpTruck;
    Mesh3D plane;
    Mesh3D hourglas;

    void initWindow(){
        window = gpu::Window("Application", WIDTH, HEIGHT);
        camera = gpu::Camera(gpu::Camera::Type::eTrackBall, window.getGLFWWindow(), WIDTH, HEIGHT, glm::vec3(0.0f, 0.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f));
        input = gpu::InputManager(window);
    }

    void toggleWireframe(){
        static bool wireframe = false;
        device.waitIdle();

        if(wireframe){
            wireframe = false;
        }
        else{
            wireframe = true;
        }
        triangleRenderPass.destroyGraphicsPipeline();
        triangleRenderPass.createGraphicsPipeline(wireframe);
    }

    void loadScene(int scene){
        // simulation.rigidBodies.clear();
        triangleRenderPass.models.clear();
        switch (scene)
        {
        case 0: // Dump truck scene
            triangleRenderPass.models.push_back(dumpTruckModel);
            triangleRenderPass.models.push_back(planeModel);

            core.updateBufferData(simulation.particlesBufferB, simulation.lrParticles2.data(), simulation.lrParticles2.size() * sizeof(LRParticle));
            core.updateBufferData(simulation.particlesBufferHR, simulation.hrParticles2.data(), simulation.hrParticles2.size() * sizeof(HRParticle));

            simulation.volumeMapTransforms[0].enable(); // enable dump_truck
            simulation.volumeMapTransforms[1].enable(); // enable plane
            simulation.volumeMapTransforms[2].disable(); // disable hourglas
            break;
        case 1: // Plane only
            triangleRenderPass.models.push_back(planeModel);

            core.updateBufferData(simulation.particlesBufferB, simulation.lrParticles.data(), simulation.lrParticles.size() * sizeof(LRParticle));
            core.updateBufferData(simulation.particlesBufferHR, simulation.hrParticles.data(), simulation.hrParticles.size() * sizeof(HRParticle));

            simulation.volumeMapTransforms[0].disable(); // disable dump_truck
            simulation.volumeMapTransforms[1].enable(); // enable plane
            simulation.volumeMapTransforms[2].disable(); // disable hourglas
            break;
        case 2: // hourglas scene
            triangleRenderPass.models.push_back(hourglasModel);

            core.updateBufferData(simulation.particlesBufferB, simulation.lrParticles2.data(), simulation.lrParticles2.size() * sizeof(LRParticle));
            core.updateBufferData(simulation.particlesBufferHR, simulation.hrParticles2.data(), simulation.hrParticles2.size() * sizeof(HRParticle));

            simulation.volumeMapTransforms[0].disable(); // disable dump_truck
            simulation.volumeMapTransforms[1].disable(); // disable plane
            simulation.volumeMapTransforms[2].enable(); // hourglas hourglas
            break;
        
        default:
            break;
        }
        simulation.updateVolumeMapTransforms();
    }


    void initVulkan(){
        core = gpu::Core(true, &window);
        
        device = core.getDevice();

        core.createComputeBundle(computeBundle);

        particleRenderPass = gpu::ParticleRenderPass(&core, &camera);
        triangleRenderPass = gpu::TriangleRenderPass(&core, &camera);
        imguiRenderPass = gpu::ImguiRenderPass(&core, &window);
        using std::placeholders::_1;
        imguiRenderPass.changeSceneCallback = std::bind(&Application::loadScene, this, _1);
        imguiRenderPass.toggleWireframeCallback = std::bind(&Application::toggleWireframe, this);

        simulation = GranularMatter(&core);

        Model::getTexturesLayout(&core);

        // Load rigidbodies for simulation
        dumpTruck = Mesh3D(ASSETS_PATH"/models/dump_truck.glb");
        plane = Mesh3D(ASSETS_PATH"/models/plane.glb");
        hourglas = Mesh3D(ASSETS_PATH"/models/hourglas.glb");

        // create signed distance fields
        simulation.rigidBodies.push_back(&dumpTruck);
        simulation.rigidBodies.push_back(&plane);
        simulation.rigidBodies.push_back(&hourglas);
        simulation.createSignedDistanceFields();
        
        simulation.init();

        // Load models
        dumpTruckModel = Model(&core);
        dumpTruckModel.load_from_glb(ASSETS_PATH "/models/dump_truck.glb");

        planeModel = Model(&core);
        planeModel.load_from_glb(ASSETS_PATH "/models/plane.glb");

        hourglasModel = Model(&core);
        hourglasModel.load_from_glb(ASSETS_PATH "/models/hourglas.glb");

        loadScene(0);
        
        particleRenderPass.vertexBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);

        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            particleRenderPass.vertexBuffer[i] = simulation.particlesBufferHR;
        }
        particleRenderPass.vertexCount = static_cast<uint32_t>(simulation.hrParticles.size());
        particleRenderPass.attributeDescriptions = HRParticle::getAttributeDescriptions();
        particleRenderPass.bindingDescription = HRParticle::getBindingDescription();

        // for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        //     particleRenderPass.vertexBuffer[i] = simulation.particlesBufferB;
        // }
        // particleRenderPass.vertexCount = static_cast<uint32_t>(simulation.lrParticles.size());
        // particleRenderPass.attributeDescriptions = LRParticle::getAttributeDescriptions();
        // particleRenderPass.bindingDescription = LRParticle::getBindingDescription();

        particleRenderPass.init();
        triangleRenderPass.init();

    }

    void drawFrame(float dt){
        size_t currentFrame = core._swapChainBundle._currentFrame;
        vk::Result result;
        result = device.waitForFences(computeBundle._frames[currentFrame]._inFlight, VK_TRUE, UINT64_MAX);

        device.resetFences(computeBundle._frames[currentFrame]._inFlight);

        simulation.update(static_cast<int>(currentFrame), 0, dt);
        
        std::array<vk::CommandBuffer, 1> submitComputeCommandBuffers = { 
            simulation.getCommandBuffer(static_cast<int>(currentFrame))
        }; 
        
        {
            std::vector<vk::Semaphore> signalComputeSemaphores = {
                computeBundle._frames[currentFrame]._computeFinished
            };

            std::vector<vk::Semaphore> waitSemaphores = {
                simulation.iisphSemaphores[currentFrame]
            };
            std::vector<vk::PipelineStageFlags> waitStages = {
                vk::PipelineStageFlagBits::eComputeShader
            };

            vk::SubmitInfo computeSubmitInfo{
                waitSemaphores,
                waitStages,
                submitComputeCommandBuffers,
                signalComputeSemaphores
            };

            core.getComputeQueue().submit(computeSubmitInfo, computeBundle._frames[currentFrame]._inFlight);
        }

        result = device.waitForFences(core.getCurrentFrame()._inFlight, VK_TRUE, UINT64_MAX);
        device.resetFences(core.getCurrentFrame()._inFlight);

        uint32_t imageIndex;
        
        vk::Result accuireNextImageResult = core.acquireNextImageKHR(&imageIndex, core.getCurrentFrame()._imageAvailable);

        if(accuireNextImageResult == vk::Result::eErrorOutOfDateKHR){
            recreateSwapChain();
            return;
        }
        else if(accuireNextImageResult != vk::Result::eSuccess && accuireNextImageResult != vk::Result::eSuboptimalKHR){
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        particleRenderPass.update(static_cast<int>(currentFrame), imageIndex, dt);
        triangleRenderPass.update(static_cast<int>(currentFrame), imageIndex, dt);
        imguiRenderPass.update(static_cast<int>(currentFrame), imageIndex, dt);

        std::vector<vk::Semaphore> waitSemaphores = {
            computeBundle._frames[currentFrame]._computeFinished, 
            core.getCurrentFrame()._imageAvailable
        };
        std::vector<vk::PipelineStageFlags> waitStages = {
            vk::PipelineStageFlagBits::eVertexInput, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };
        std::vector<vk::Semaphore> signalSemaphores = {
            core.getCurrentFrame()._renderFinished
        };
        std::array<vk::CommandBuffer, 3> submitCommandBuffers = { 
            particleRenderPass.getCommandBuffer(static_cast<int>(currentFrame)), 
            triangleRenderPass.getCommandBuffer(static_cast<int>(currentFrame)), 
            imguiRenderPass.getCommandBuffer(static_cast<int>(currentFrame))
        };
        vk::SubmitInfo submitInfo(waitSemaphores, waitStages, submitCommandBuffers, signalSemaphores);

        core.getGraphicsQueue().submit(submitInfo, core.getCurrentFrame()._inFlight);

        vk::Result presentResult = core.presentKHR(imageIndex, signalSemaphores);

        if(presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR || window.wasResized()){
            window.resizeHandled();
            recreateSwapChain();
        }
        else if(presentResult != vk::Result::eSuccess){
            throw std::runtime_error("queue Present failed!");
        }
        core._swapChainBundle._currentFrame = (currentFrame + 1) % gpu::MAX_FRAMES_IN_FLIGHT;
    }

    void mainLoop(){
        
        std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
        float accumulatedTime = 0;
        uint32_t fps = 0;

        while (!window.shouldClose()) {

            auto currentTime = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
            startTime = std::chrono::high_resolution_clock::now();

            input.update();
            glfwPollEvents();

            camera.update(dt);

            drawFrame(dt);
            
            fps++;
            accumulatedTime += dt;
            if(accumulatedTime >= 1.0){
                std::string title = "Application  FPS:"+std::to_string(fps);
                window.setTitle(title);
                fps = 0;
                accumulatedTime = 0;
            }
        }
        device.waitIdle();
    }

    void cleanupSwapchain(){
        core.getDevice().waitIdle();
        core.destroySwapChain(); 
    }

    void cleanup(){
        core.getDevice().waitIdle();

        
        particleRenderPass.destroy();
        triangleRenderPass.destroy();
        imguiRenderPass.destroy();


        hourglasModel.destroy();
        dumpTruckModel.destroy();
        planeModel.destroy();

        simulation.destroy();

        Model::cleanupDescriptorSetLayouts(&core);

        cleanupSwapchain();

        core.destroyComputeBundle(computeBundle);
        
        window.destroy();
    }

    void recreateSwapChain() {
        int width = 0, height = 0;
        window.getSize(&width, &height);
        while (width == 0 || height == 0) {
            window.getSize(&width, &height);
            glfwWaitEvents();
        }
        device.waitIdle();

        particleRenderPass.destroyFrameResources();
        triangleRenderPass.destroyFrameResources();
        imguiRenderPass.destroyFrameResources();
        simulation.destroyFrameResources();

        cleanupSwapchain();

        core.createSwapChain(&window);
   
        particleRenderPass.initFrameResources();
        triangleRenderPass.initFrameResources();
        imguiRenderPass.initFrameResources();
        simulation.initFrameResources();

    }
};

int main() {
    Application app;
    try {
        app.run();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}