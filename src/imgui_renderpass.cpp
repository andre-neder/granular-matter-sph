#include "imgui_renderpass.h"
#include "global.h"
#include <glm/gtc/type_ptr.hpp>
bool simulationRunning = false;

SPHSettings settings = SPHSettings();
bool show_demo_window = false;
bool showGPUInfo = true;
bool showSimulationSettings = true;

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
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();
        ImGui_ImplGlfw_InitForVulkan(window->getGLFWWindow(), true);

        createDescriptorPool();
        createRenderPass();      

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

        initFrameResources(); // Todo: fix cleanup error
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

        static bool p_dockSpaceOpen = true;
        static bool opt_fullscreen = true;
        static bool opt_padding = false;
        static ImGuiDockNodeFlags dockspace_flags = ImGuiDockNodeFlags_PassthruCentralNode;

        // We are using the ImGuiWindowFlags_NoDocking flag to make the parent window not dockable into,
        // because it would be confusing to have two docking targets within each others.
        ImGuiWindowFlags window_flags = ImGuiWindowFlags_MenuBar | ImGuiWindowFlags_NoDocking;
        if (opt_fullscreen)
        {
            const ImGuiViewport* viewport = ImGui::GetMainViewport();
            ImGui::SetNextWindowPos(viewport->WorkPos);
            ImGui::SetNextWindowSize(viewport->WorkSize);
            ImGui::SetNextWindowViewport(viewport->ID);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
            ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
            window_flags |= ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove;
            window_flags |= ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;
        }
        else
        {
            dockspace_flags &= ~ImGuiDockNodeFlags_PassthruCentralNode;
        }

        // When using ImGuiDockNodeFlags_PassthruCentralNode, DockSpace() will render our background
        // and handle the pass-thru hole, so we ask Begin() to not render a background.
        if (dockspace_flags & ImGuiDockNodeFlags_PassthruCentralNode)
            window_flags |= ImGuiWindowFlags_NoBackground;

        // Important: note that we proceed even if Begin() returns false (aka window is collapsed).
        // This is because we want to keep our DockSpace() active. If a DockSpace() is inactive,
        // all active windows docked into it will lose their parent and become undocked.
        // We cannot preserve the docking relationship between an active window and an inactive docking, otherwise
        // any change of dockspace/settings would lead to windows being stuck in limbo and never being visible.
        if (!opt_padding)
            ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(0.0f, 0.0f));
        ImGui::Begin("DockSpace Demo", &p_dockSpaceOpen, window_flags);
        if (!opt_padding)
            ImGui::PopStyleVar();

        if (opt_fullscreen)
            ImGui::PopStyleVar(2);

        // Submit the DockSpace
        ImGuiIO& io = ImGui::GetIO();
        if (io.ConfigFlags & ImGuiConfigFlags_DockingEnable)
        {
            ImGuiID dockspace_id = ImGui::GetID("MyDockSpace");
            ImGui::DockSpace(dockspace_id, ImVec2(0.0f, 0.0f), dockspace_flags);
        }

            ImGui::Begin("GPU Info", &showGPUInfo);
                ImGui::Text(m_deviceProperties.deviceName);
            ImGui::End();

            ImGui::Begin("Timings", &showGPUInfo);
                for (size_t i = 0; i < passTimeings.size(); i++) {
                    ImGui::Text(passTimeings[i].c_str());
                }
                passTimeings = std::vector<std::string>();
            ImGui::End();

            ImGui::Begin("Simulation", &showSimulationSettings);
                ImGui::Checkbox("Simulation running", &simulationRunning);
                ImGui::DragFloat("Rest Density (kg/m^2)", &settings.rho0, 1.f, 0.1f, 2000.f);
                ImGui::DragFloat("Pressure stiffness", &settings.stiffness, 1.f, 100.f, 50000.f);
                ImGui::DragFloat("Mass (kg)", &settings.mass, 1.f, 0.1f, 100.f);
                ImGui::DragFloat("Kernel Radius (m)", &settings.kernelRadius, 1.f, 0.1f, 100.f);
                ImGui::DragFloat("Angle of repose (rad)", &settings.theta, 1.f, 0.001f, (float)M_PI);
                ImGui::DragFloat("Viscosity constant", &settings.sigma, 1.f, 0.01f, 10.f);
                // ImGui::DragFloat("Cohesion intensity", &settings.beta, 1.f, 0.01f, 10.f);
                // ImGui::DragFloat("Maximum Cohesion", &settings.C, 1.f, 0.01f, 10.f);
                ImGui::DragFloat2("Gravity (m/s^2)", glm::value_ptr(settings.g));
            ImGui::End();

            ImGui::ShowDemoWindow(&show_demo_window);

        ImGui::End();

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
        descriptorPool = m_core->createDescriptorPool({
            { vk::DescriptorType::eSampler, 1000 },
            { vk::DescriptorType::eCombinedImageSampler, 1000 },
            { vk::DescriptorType::eSampledImage, 1000 },
            { vk::DescriptorType::eStorageImage, 1000 },
            { vk::DescriptorType::eUniformTexelBuffer, 1000 },
            { vk::DescriptorType::eStorageTexelBuffer, 1000 },
            { vk::DescriptorType::eUniformBuffer, 1000 },
            { vk::DescriptorType::eStorageBuffer, 1000 },
            { vk::DescriptorType::eUniformBufferDynamic, 1000 },
            { vk::DescriptorType::eStorageBufferDynamic, 1000 },
            { vk::DescriptorType::eInputAttachment, 1000 }
        });
    }
    
    void ImguiRenderPass::destroyFrameResources(){
        vk::Device device = m_core->getDevice();
        for (auto framebuffer : framebuffers) {
            device.destroyFramebuffer(framebuffer);
        }
        device.freeCommandBuffers(m_core->getCommandPool(), commandBuffers);
        m_core->destroyDescriptorPool(descriptorPool);
        m_core->getDevice().destroyRenderPass(renderPass);
    }
    void ImguiRenderPass::destroy(){
        destroyFrameResources();

        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImGui::DestroyContext();
    }
    
