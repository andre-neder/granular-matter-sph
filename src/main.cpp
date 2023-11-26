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

#include "screenquad_renderpass.h"
#include "imgui_renderpass.h"

#include "global.h"

std::vector<std::string> passTimeings = std::vector<std::string>();

const uint32_t WIDTH = 1920;
const uint32_t HEIGHT = 1080;

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

    gpu::ScreenQuadRenderPass screenQuadPass;
    gpu::ImguiRenderPass imguiRenderPass;

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

        screenQuadPass = gpu::ScreenQuadRenderPass(&core);
        imguiRenderPass = gpu::ImguiRenderPass(&core, &window);
        screenQuadPass.init();
        createSyncObjects();
    }

    void createSyncObjects() {
        imageAvailableSemaphores.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(gpu::MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(gpu::MAX_FRAMES_IN_FLIGHT, VK_NULL_HANDLE);
        vk::SemaphoreCreateInfo semaphoreInfo;
        vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);
        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            try{
                imageAvailableSemaphores[i] = device.createSemaphore(semaphoreInfo);
                renderFinishedSemaphores[i] = device.createSemaphore(semaphoreInfo);
                inFlightFences[i] = device.createFence(fenceInfo);
            }catch(std::exception& e) {
                std::cerr << "Exception Thrown: " << e.what();
            }
        }
    }

    void drawFrame(){
        vk::Result result;
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

        screenQuadPass.update((int) currentFrame, imageIndex);
        imguiRenderPass.update((int) currentFrame, imageIndex);


        if ((VkFence) imagesInFlight[currentFrame] != VK_NULL_HANDLE){
            vk::Result result2 = device.waitForFences(imagesInFlight[currentFrame], VK_TRUE, UINT64_MAX);
        }
        imagesInFlight[currentFrame] = inFlightFences[currentFrame];

        std::vector<vk::Semaphore> waitSemaphores = {
            imageAvailableSemaphores[currentFrame]
        };
        std::vector<vk::PipelineStageFlags> waitStages = {
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };
        std::vector<vk::Semaphore> signalSemaphores = {
            renderFinishedSemaphores[currentFrame]
        };
        std::array<vk::CommandBuffer, 2> submitCommandBuffers = { 
            screenQuadPass.getCommandBuffer((int) currentFrame),
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
        while (!window.shouldClose()) {
            glfwPollEvents();

            drawFrame();
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
        
        screenQuadPass.destroy();
        imguiRenderPass.destroy();

        cleanupSwapchain();

        for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
            device.destroySemaphore(imageAvailableSemaphores[i]);
            device.destroySemaphore(renderFinishedSemaphores[i]);
            device.destroyFence(inFlightFences[i]);
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

        screenQuadPass.destroyFrameResources();
        imguiRenderPass.destroyFrameResources();
        
        cleanupSwapchain();

        core.createSwapChain(&window);
        core.createSwapChainImageViews();
   
        screenQuadPass.initFrameResources();
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