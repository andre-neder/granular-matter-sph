#pragma once

#define GLM_FORCE_RADIANS
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

#include "basic_renderpass.h"
#include "imgui_renderpass.h"
#include "line_renderpass.h"
#include "granular_matter.h"

#include "global.h"

std::vector<std::string> passTimeings = std::vector<std::string>();

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
    bool enableValidation = true;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;

    gpu::Core core;
    gpu::Window window;

    gpu::BasicRenderPass basicRenderPass;
    gpu::LineRenderPass lineRenderPass;
    gpu::ImguiRenderPass imguiRenderPass;

    GranularMatter simulation;


    std::vector<vk::Fence> computeInFlightFences;
    std::vector<vk::Semaphore> computeFinishedSemaphores;
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphores;
    std::vector<vk::Fence> inFlightFences;
    std::vector<vk::Fence> imagesInFlight;
    size_t currentFrame = 0;

    void initWindow(){
        window = gpu::Window("Application", WIDTH, HEIGHT);
    }

    void initVulkan(){
        core = gpu::Core(enableValidation, &window);
        physicalDevice = core.getPhysicalDevice();
        device = core.getDevice();

        basicRenderPass = gpu::BasicRenderPass(&core);
        lineRenderPass = gpu::LineRenderPass(&core);
        imguiRenderPass = gpu::ImguiRenderPass(&core, &window);

        simulation = GranularMatter(&core);

        basicRenderPass.vertexBuffer.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            basicRenderPass.vertexBuffer[i] = simulation.particlesBufferB[i];
        }
        
        basicRenderPass.vertexCount = (uint32_t)simulation.particles.size();

        basicRenderPass.attributeDescriptions = HRParticle::getAttributeDescriptions();
        basicRenderPass.bindingDescription = HRParticle::getBindingDescription();

        basicRenderPass.init();
        lineRenderPass.init();

        createSyncObjects();
    }

    void createSyncObjects() {
        std::cout << "MAX_FRAMES_IN_FLIGHT: " << gpu::MAX_FRAMES_IN_FLIGHT << std::endl;
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

    void drawFrame(){
        vk::Result result;
        result = device.waitForFences(computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        device.resetFences(computeInFlightFences[currentFrame]);

        simulation.update((int)currentFrame, 0);
        
        std::array<vk::CommandBuffer, 1> submitComputeCommandBuffers = { 
            simulation.getCommandBuffer((int)currentFrame)
        }; 

        std::vector<vk::Semaphore> signalComputeSemaphores = {computeFinishedSemaphores[currentFrame]};

        vk::SubmitInfo computeSubmitInfo{
            {},
            {},
            submitComputeCommandBuffers,
            signalComputeSemaphores
        };

        core.getComputeQueue().submit(computeSubmitInfo, computeInFlightFences[currentFrame]);


        result = device.waitForFences(inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        uint32_t imageIndex;
        
        try{
            result = device.acquireNextImageKHR(core.getSwapChain(), UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        }
        catch(const std::exception& e){
            std::cerr << e.what() << '\n';
        }
        if(result == vk::Result::eErrorOutOfDateKHR){
            recreateSwapChain();
            return;
        }
        else if(result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR){
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        basicRenderPass.update((int) currentFrame, imageIndex);
        lineRenderPass.update((int) currentFrame, imageIndex);
        imguiRenderPass.update((int) currentFrame, imageIndex);


        if ((VkFence) imagesInFlight[currentFrame] != VK_NULL_HANDLE){
            vk::Result result2 = device.waitForFences(imagesInFlight[currentFrame], VK_TRUE, UINT64_MAX);
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
            basicRenderPass.getCommandBuffer((int) currentFrame), 
            lineRenderPass.getCommandBuffer((int) currentFrame), 
            imguiRenderPass.getCommandBuffer((int) currentFrame)
        };
        vk::SubmitInfo submitInfo(waitSemaphores, waitStages, submitCommandBuffers, signalSemaphores);

        device.resetFences(inFlightFences[currentFrame]);

        core.getGraphicsQueue().submit(submitInfo, inFlightFences[currentFrame]);

        std::vector<vk::SwapchainKHR> swapChains = {core.getSwapChain()};
        vk::PresentInfoKHR presentInfo(signalSemaphores, swapChains, imageIndex);
        try{
            result = core.getPresentQueue().presentKHR(presentInfo);
        }
        catch(const std::exception& e){
            std::cerr << e.what() << '\n';
        }
        
        if(result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || window.wasResized()){
            window.resizeHandled();
            recreateSwapChain();
        }else if(result != vk::Result::eSuccess)
            throw std::runtime_error("queue Present failed!");
        currentFrame = (currentFrame + 1) % gpu::MAX_FRAMES_IN_FLIGHT;
    }

    void mainLoop(){
        double time = glfwGetTime();
        uint32_t fps = 0;
        while (!window.shouldClose()) {
            glfwPollEvents();

            drawFrame();
            
            fps++;
            if((glfwGetTime() - time) >= 1.0){
                time = glfwGetTime();
                std::string title = "Application  FPS:"+std::to_string(fps);
                window.setTitle(title);
                fps = 0;
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

        
        basicRenderPass.destroy();
        lineRenderPass.destroy();
        imguiRenderPass.destroy();

        simulation.destroy();

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

        basicRenderPass.destroyFrameResources();
        lineRenderPass.destroyFrameResources();
        imguiRenderPass.destroyFrameResources();
        cleanupSwapchain();

        core.createSwapChain(&window);
        core.createSwapChainImageViews();
   
        basicRenderPass.initFrameResources();
        lineRenderPass.initFrameResources();
        imguiRenderPass.initFrameResources();

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