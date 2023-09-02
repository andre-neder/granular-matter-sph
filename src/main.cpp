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
// #include "granular_matter.h"


const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;
const int MAX_FRAMES_IN_FLIGHT = 2;

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
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    vk::SurfaceKHR surface;

    vk::Extent2D swapChainExtent;

    gpu::BasicRenderPass basicRenderPass;
    gpu::ImguiRenderPass imguiRenderPass;

    // GranularMatter simulation;

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
        surface = core.getSurface();
        physicalDevice = core.getPhysicalDevice();
        device = core.getDevice();
        graphicsQueue = core.getGraphicsQueue();
        presentQueue = core.getPresentQueue();
        swapChainExtent = core.getSwapChainExtent();

        basicRenderPass = gpu::BasicRenderPass(&core);
        imguiRenderPass = gpu::ImguiRenderPass(&core, &window);

        // simulation = GranularMatter(&core);
        
        createSyncObjects();
    }

    void createSyncObjects() {
        computeInFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        computeFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        renderFinishedSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
        inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
        imagesInFlight.resize(core.getSwapChainImageCount(), VK_NULL_HANDLE);
        vk::SemaphoreCreateInfo semaphoreInfo;
        vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);
        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
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

        // vk::Result result2 = device.waitForFences(computeInFlightFences[currentFrame], VK_TRUE, UINT64_MAX);

        // device.resetFences(computeInFlightFences[currentFrame]);

        // simulation.update(currentFrame);
        
        // std::array<vk::CommandBuffer, 1> submitComputeCommandBuffers = { 
        //     simulation.getCommandBuffer(currentFrame)
        // }; 

        // std::vector<vk::Semaphore> signalComputeSemaphores = {computeFinishedSemaphores[currentFrame]};

        // vk::SubmitInfo computeSubmitInfo{
        //     {},
        //     {},
        //     submitComputeCommandBuffers,
        //     signalComputeSemaphores
        // };


        // core.getComputeQueue().submit(computeSubmitInfo);


        vk::Result result1 = device.waitForFences(inFlightFences[currentFrame], VK_TRUE, UINT64_MAX);
        uint32_t imageIndex;
        vk::Result result;
        try{
            result = device.acquireNextImageKHR(core.getSwapChain(), UINT64_MAX, imageAvailableSemaphores[currentFrame], VK_NULL_HANDLE, &imageIndex);
        }
        catch(const std::exception& e){
            // std::cerr << e.what() << '\n';
        }
        if(result == vk::Result::eErrorOutOfDateKHR){
            recreateSwapChain();
            return;
        }
        else if(result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR){
            throw std::runtime_error("failed to acquire swap chain image!");
        }

        // updateUniformBuffer(imageIndex);
        recordCommandBuffer(imageIndex);


        if ((VkFence) imagesInFlight[imageIndex] != VK_NULL_HANDLE){
            vk::Result result2 = device.waitForFences(imagesInFlight[imageIndex], VK_TRUE, UINT64_MAX);
        }
        imagesInFlight[imageIndex] = inFlightFences[currentFrame];

        std::vector<vk::Semaphore> waitSemaphores = {
            // computeFinishedSemaphores[currentFrame], 
            imageAvailableSemaphores[currentFrame]
        };
        std::vector<vk::PipelineStageFlags> waitStages = {
            // vk::PipelineStageFlagBits::eVertexInput, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput
        };
        std::vector<vk::Semaphore> signalSemaphores = {
            renderFinishedSemaphores[currentFrame]
        };
        std::array<vk::CommandBuffer, 2> submitCommandBuffers = { 
            basicRenderPass.getCommandBuffer(imageIndex), 
            imguiRenderPass.getCommandBuffer(imageIndex)
        };
        vk::SubmitInfo submitInfo(waitSemaphores, waitStages, submitCommandBuffers, signalSemaphores);

        device.resetFences(inFlightFences[currentFrame]);

        graphicsQueue.submit(submitInfo, inFlightFences[currentFrame]);

        std::vector<vk::SwapchainKHR> swapChains = {core.getSwapChain()};
        vk::PresentInfoKHR presentInfo(signalSemaphores, swapChains, imageIndex);
        try{
            result = presentQueue.presentKHR(presentInfo);
        }
        catch(const std::exception& e){
            // std::cerr << e.what() << '\n';
        }
        
        if(result == vk::Result::eErrorOutOfDateKHR || result == vk::Result::eSuboptimalKHR || window.wasResized()){
            // framebufferResized = false;
            window.resizeHandled();
            recreateSwapChain();
        }else if(result != vk::Result::eSuccess)
            throw std::runtime_error("queue Present failed!");
        currentFrame = (currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;
    }

    void recordCommandBuffer(uint32_t imageIndex){
        
        imguiRenderPass.update(imageIndex);
        basicRenderPass.update(imageIndex);
    
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
        basicRenderPass.destroy();
        imguiRenderPass.destroy();

        // simulation.destroy();

        cleanupSwapchain();

        for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
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
        imguiRenderPass.destroyFrameResources();
        cleanupSwapchain();

        core.createSwapChain(&window);
        swapChainExtent = core.getSwapChainExtent();
        core.createSwapChainImageViews();
   
        basicRenderPass.initFrameResources();
        imguiRenderPass.initFrameResources();

        imagesInFlight.resize(core.getSwapChainImageCount(), VK_NULL_HANDLE);

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