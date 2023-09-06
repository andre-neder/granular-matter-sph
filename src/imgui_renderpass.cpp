#include "imgui_renderpass.h"

using namespace gpu;

    ImguiRenderPass::ImguiRenderPass(gpu::Core* core, gpu::Window* window){
        m_core = core;
        m_deviceProperties = m_core->getPhysicalDevice().getProperties();
        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
        //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();
        ImGui_ImplGlfw_InitForVulkan(window->getGLFWWindow(), true);

        initFrameResources();
  
        ImGui_ImplVulkan_InitInfo init_info = {};
        init_info.Instance = m_core->getInstance();
        init_info.PhysicalDevice = m_core->getPhysicalDevice();
        init_info.Device = m_core->getDevice();
        init_info.QueueFamily = m_core->findQueueFamilies(m_core->getPhysicalDevice()).graphicsFamily.value();
        init_info.Queue = m_core->getGraphicsQueue();
        init_info.PipelineCache = VK_NULL_HANDLE;
        init_info.DescriptorPool = descriptorPool;
        init_info.Allocator = VK_NULL_HANDLE;
        init_info.MinImageCount = static_cast<uint32_t>(m_core->getSwapChainImageCount());
        init_info.ImageCount = static_cast<uint32_t>(m_core->getSwapChainImageCount());
        init_info.CheckVkResultFn = check_vk_result;
        ImGui_ImplVulkan_Init(&init_info, renderPass);

        vk::CommandBuffer command_buffer = m_core->beginSingleTimeCommands();
        ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
        m_core->endSingleTimeCommands(command_buffer);    

    }
    void ImguiRenderPass::initFrameResources(){
        createRenderPass();
        createFramebuffers();
        createDescriptorPool();
        createCommandBuffers();
    }
    void ImguiRenderPass::createRenderPass(){
       vk::AttachmentDescription attachment(
            {}, 
            m_core->getSwapChainImageFormat(), 
            vk::SampleCountFlagBits::e1, 
            vk::AttachmentLoadOp::eLoad, 
            vk::AttachmentStoreOp::eStore, 
            vk::AttachmentLoadOp::eDontCare, 
            vk::AttachmentStoreOp::eDontCare, 
            vk::ImageLayout::eColorAttachmentOptimal, 
            vk::ImageLayout::ePresentSrcKHR
        );
        vk::AttachmentReference attachmentRef(0, vk::ImageLayout::eColorAttachmentOptimal);

        vk::SubpassDescription subpass({}, vk::PipelineBindPoint::eGraphics, {}, attachmentRef, {}, {});

        vk::SubpassDependency dependency(
            VK_SUBPASS_EXTERNAL, 
            0, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput, 
            vk::PipelineStageFlagBits::eColorAttachmentOutput, 
            vk::AccessFlagBits::eNoneKHR, 
            vk::AccessFlagBits::eColorAttachmentWrite);

        std::array<vk::AttachmentDescription, 1> attachments = {attachment};

        vk::RenderPassCreateInfo renderPassInfo(
            {}, 
            attachments, 
            subpass, 
            dependency
        );
        try{
            renderPass = m_core->getDevice().createRenderPass(renderPassInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    
    void ImguiRenderPass::update(int currentFrame, int imageIndex){
        
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("GPU Info", &showGPUInfo);
        ImGui::Text(m_deviceProperties.deviceName);
        ImGui::End();

        ImGui::ShowDemoWindow(&show_demo_window);
        ImGui::Render();

        vk::CommandBufferBeginInfo beginInfo;
        try{
            commandBuffers[currentFrame].begin(beginInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
        std::array<vk::ClearValue, 1> clearValues{
            vk::ClearValue(vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f}))
        };
        vk::RenderPassBeginInfo renderPassInfo(renderPass, framebuffers[imageIndex], vk::Rect2D({0, 0}, m_core->getSwapChainExtent()), clearValues);

        commandBuffers[currentFrame].beginRenderPass(renderPassInfo, vk::SubpassContents::eInline);

        ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), commandBuffers[currentFrame]);

        commandBuffers[currentFrame].endRenderPass();
        try{
            commandBuffers[currentFrame].end();
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
   
    void ImguiRenderPass::createDescriptorPool() {
        std::array<vk::DescriptorPoolSize, 11> poolSizes{
            vk::DescriptorPoolSize{ vk::DescriptorType::eSampler, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eCombinedImageSampler, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eSampledImage, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageImage, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformTexelBuffer, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageTexelBuffer, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBuffer, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBuffer, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eUniformBufferDynamic, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eStorageBufferDynamic, 1000 },
            vk::DescriptorPoolSize{ vk::DescriptorType::eInputAttachment, 1000 }
        };
        //vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet
        vk::DescriptorPoolCreateInfo descriptorPoolInfo({}, static_cast<uint32_t>(1000 * poolSizes.size()), poolSizes);

        try{
            descriptorPool = m_core->getDevice().createDescriptorPool(descriptorPoolInfo);
        }catch(std::exception& e) {
            std::cerr << "Exception Thrown: " << e.what();
        }
    }
    
    void ImguiRenderPass::destroyFrameResources(){
        vk::Device device = m_core->getDevice();
        for (auto framebuffer : framebuffers) {
            device.destroyFramebuffer(framebuffer);
        }
        device.freeCommandBuffers(m_core->getCommandPool(), commandBuffers);
        m_core->getDevice().destroyDescriptorPool(descriptorPool);
        m_core->getDevice().destroyRenderPass(renderPass);
    }
    void ImguiRenderPass::destroy(){
        destroyFrameResources();

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }
    
