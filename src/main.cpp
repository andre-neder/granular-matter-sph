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
    vk::PhysicalDevice physicalDevice;
    vk::Device device;

    gpu::Core core;
    gpu::Camera camera;
    gpu::Window window;

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


    std::vector<vk::Fence> computeInFlightFences;
    std::vector<vk::Semaphore> computeFinishedSemaphores;
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    size_t currentFrame = 0;

    void initWindow(){
        window = gpu::Window("Application", WIDTH, HEIGHT);
        camera = gpu::Camera(gpu::Camera::Type::eTrackBall, window.getGLFWWindow(), WIDTH, HEIGHT, glm::vec3(0.0f, 0.0f, 4.0f), glm::vec3(0.0f, 0.0f, 0.0f));
    }

    
    void loadScene(int scene){
        // simulation.rigidBodies.clear();
        triangleRenderPass.models.clear();
        switch (scene)
        {
        case 0: // Dump truck scene
            triangleRenderPass.models.push_back(dumpTruckModel);
            triangleRenderPass.models.push_back(planeModel);

            simulation.volumeMapTransforms[0].enable(); // enable dump_truck
            simulation.volumeMapTransforms[1].enable(); // enable plane
            simulation.volumeMapTransforms[2].disable(); // disable hourglas
            break;
        case 1: // Plane only
            triangleRenderPass.models.push_back(planeModel);

            simulation.volumeMapTransforms[0].disable(); // disable dump_truck
            simulation.volumeMapTransforms[1].enable(); // enable plane
            simulation.volumeMapTransforms[2].disable(); // disable hourglas
            break;
        case 2: // hourglas scene
            triangleRenderPass.models.push_back(hourglasModel);

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
        physicalDevice = core.getPhysicalDevice();
        device = core.getDevice();

        particleRenderPass = gpu::ParticleRenderPass(&core, &camera);
        triangleRenderPass = gpu::TriangleRenderPass(&core, &camera);
        imguiRenderPass = gpu::ImguiRenderPass(&core, &window);
        using std::placeholders::_1;
        imguiRenderPass.changeSceneCallback = std::bind(&Application::loadScene, this, _1);

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
        particleRenderPass.vertexCount = (uint32_t)simulation.hrParticles.size();
        particleRenderPass.attributeDescriptions = HRParticle::getAttributeDescriptions();
        particleRenderPass.bindingDescription = HRParticle::getBindingDescription();

        // for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        //     particleRenderPass.vertexBuffer[i] = simulation.particlesBufferB;
        // }
        // particleRenderPass.vertexCount = (uint32_t)simulation.lrParticles.size();
        // particleRenderPass.attributeDescriptions = LRParticle::getAttributeDescriptions();
        // particleRenderPass.bindingDescription = LRParticle::getBindingDescription();

        particleRenderPass.init();
        triangleRenderPass.init();

        createSyncObjects();
    }

    void createSyncObjects() {
        computeInFlightFences.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        computeFinishedSemaphores.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        imageAvailableSemaphores.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(gpu::MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE);
        vk::SemaphoreCreateInfo semaphoreInfo;
        vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);
        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            try{
                computeFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
                imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
                renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
                inFlightFences[i] = device.createFence(fenceInfo);
                computeInFlightFences[i] = device.createFence(fenceInfo);
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
        }
    }

    void drawFrame(float dt){
        device.waitForFences(computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        device.resetFences(computeInFlightFences[currentFrame]);

        simulation.update((int)currentFrame, 0, dt);
        
        std::array<vk::CommandBuffer, 1> submitComputeCommandBuffers = { 
            simulation.getCommandBuffer((int)currentFrame)
        }; 
        
        {
            std::vector<vk::Semaphore> signalComputeSemaphores = {
                computeFinishedSemaphores[currentFrame]
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

            core.getComputeQueue().submit(computeSubmitInfo, computeInFlightFences[currentFrame]);
        }

        device.waitForFences(inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        
        vk::Result accuireNextImageResult;

        try{
            accuireNextImageResult = device.acquireNextImageKHR(core.getSwapChain(), UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        }
        catch(const vk::OutOfDateKHRError outOfDateError){
            accuireNextImageResult = vk::Result::eErrorOutOfDateKHR;
        }
        catch(const std::exception& e){
            std::cerr << e.what() << '\n';
        }

        if(accuireNextImageResult == vk::Result::eErrorOutOfDateKHR){
            recreateSwapChain();
            return;
        }
        else if(accuireNextImageResult != vk::Result::eSuccess && accuireNextImageResult != vk::Result::eSuboptimalKHR){
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        particleRenderPass.update((int) currentFrame, imageIndex, dt);
        triangleRenderPass.update((int) currentFrame, imageIndex, dt);
        imguiRenderPass.update((int) currentFrame, imageIndex, dt);


        if ((VkFence) imagesInFlight[currentFrame] != VK_NULL_HANDLE){
            device.waitForFences(imagesInFlight[currentFrame], VK_TRUE, UINT64_MAX);
        }
        imagesInFlight[currentFrame] = inFlightFences[currentFrame];

        std::vector<vk::Semaphore> waitSemaphores = {
            computeFinishedSemaphores[currentFrame], 
            imageAvailableSemaphores[currentFrame]
        };
        std::vector<vk::PipelineStageFlags> waitStages = {
            vk::PipelineStageFlagBits::eVertexInput, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };
        std::vector<vk::Semaphore> signalSemaphores = {
            renderFinishedSemaphores[currentFrame]
        };
        std::array<vk::CommandBuffer, 3> submitCommandBuffers = { 
            particleRenderPass.getCommandBuffer((int) currentFrame), 
            triangleRenderPass.getCommandBuffer((int) currentFrame), 
            imguiRenderPass.getCommandBuffer((int) currentFrame)
        };
        vk::SubmitInfo submitInfo(waitSemaphores, waitStages, submitCommandBuffers, signalSemaphores);

        device.resetFences(inFlightFences[currentFrame]);

        core.getGraphicsQueue().submit(submitInfo, inFlightFences[currentFrame]);

        std::vector<vk::SwapchainKHR> swapChains = {core.getSwapChain()};
        vk::PresentInfoKHR presentInfo(signalSemaphores, swapChains, imageIndex);
        vk::Result presentResult;
        try{
            presentResult = core.getPresentQueue().presentKHR(presentInfo);
        }
        catch(const vk::OutOfDateKHRError outOfDateError){
            presentResult = vk::Result::eErrorOutOfDateKHR;
        }
        catch(const std::exception& e){
            std::cerr << e.what() << '\n';
        }
        
        if(presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR || window.wasResized()){
            window.resizeHandled();
            recreateSwapChain();
        }
        else if(presentResult != vk::Result::eSuccess){
            throw std::runtime_error("queue Present failed!");
        }
        currentFrame = (currentFrame + 1) % gpu::MAX_FRAMES_IN_FLIGHT;
    }

    void mainLoop(){
        
        std::chrono::high_resolution_clock::time_point startTime = std::chrono::high_resolution_clock::now();
        float accumulatedTime = 0;
        uint32_t fps = 0;

        while (!window.shouldClose()) {

            auto currentTime = std::chrono::high_resolution_clock::now();
            float dt = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();
            startTime = std::chrono::high_resolution_clock::now();

            glfwPollEvents();

            camera.handleInput();
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
        core.destroySwapChainImageViews();
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

        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            device.destroySemaphore(imageAvailableSemaphores[i]);
            device.destroySemaphore(renderFinishedSemaphores[i]);
            device.destroySemaphore(computeFinishedSemaphores[i]);
            device.destroyFence(inFlightFences[i]);
            device.destroyFence(computeInFlightFences[i]);
        }
        
        window.destroy();

        core.destroy();
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
        core.createSwapChainImageViews();
   
        particleRenderPass.initFrameResources();
        triangleRenderPass.initFrameResources();
        imguiRenderPass.initFrameResources();
        simulation.initFrameResources();


        imagesInFlight.resize(gpu::MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE);

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