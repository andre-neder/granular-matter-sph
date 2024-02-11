#include "imgui_renderpass.h"
#include "global.h"
#include <glm/gtc/type_ptr.hpp>
#include "input.h"
#include <cctype>

bool simulationRunning = false;
bool resetSimulation = false;
int pauseOnFrame = -1;
int currentFrameCount = 0;
float simulationSpeedFactor = 1.0;

SPHSettings settings = SPHSettings();
extern std::vector<std::vector<std::string>> timestampLabels;
extern std::vector<std::vector<uint64_t>> timestamps;

bool show_demo_window = false;
bool showGPUInfo = true;
bool showSimulationSettings = true;

using namespace gpu;

ImguiRenderPass::ImguiRenderPass(gpu::Core* core, gpu::Window* window){
    _core = core;
    m_window = window;

    createDescriptorPool();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;

    io.FontDefault = io.Fonts->AddFontFromFileTTF(ASSETS_PATH"/fonts/Open_Sans/static/OpenSans-Regular.ttf", 18.f);
  
    ImGui::StyleColorsDark();
    // https://github.com/ocornut/imgui/issues/707
    auto& colors = ImGui::GetStyle().Colors;

    colors[ImGuiCol_Text]                   = ImVec4(1.00f, 1.00f, 1.00f, 1.00f);
    colors[ImGuiCol_TextDisabled]           = ImVec4(0.50f, 0.50f, 0.50f, 1.00f);
    colors[ImGuiCol_WindowBg]               = ImVec4(0.10f, 0.10f, 0.10f, 1.00f);
    colors[ImGuiCol_ChildBg]                = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_PopupBg]                = ImVec4(0.19f, 0.19f, 0.19f, 0.92f);
    colors[ImGuiCol_Border]                 = ImVec4(0.19f, 0.19f, 0.19f, 0.29f);
    colors[ImGuiCol_BorderShadow]           = ImVec4(0.00f, 0.00f, 0.00f, 0.24f);
    colors[ImGuiCol_FrameBg]                = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_FrameBgHovered]         = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
    colors[ImGuiCol_FrameBgActive]          = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_TitleBg]                = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TitleBgActive]          = ImVec4(0.06f, 0.06f, 0.06f, 1.00f);
    colors[ImGuiCol_TitleBgCollapsed]       = ImVec4(0.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_MenuBarBg]              = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_ScrollbarBg]            = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_ScrollbarGrab]          = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabHovered]   = ImVec4(0.40f, 0.40f, 0.40f, 0.54f);
    colors[ImGuiCol_ScrollbarGrabActive]    = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
    colors[ImGuiCol_CheckMark]              = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_SliderGrab]             = ImVec4(0.34f, 0.34f, 0.34f, 0.54f);
    colors[ImGuiCol_SliderGrabActive]       = ImVec4(0.56f, 0.56f, 0.56f, 0.54f);
    colors[ImGuiCol_Button]                 = ImVec4(0.05f, 0.05f, 0.05f, 0.54f);
    colors[ImGuiCol_ButtonHovered]          = ImVec4(0.19f, 0.19f, 0.19f, 0.54f);
    colors[ImGuiCol_ButtonActive]           = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_Header]                 = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_HeaderHovered]          = ImVec4(0.00f, 0.00f, 0.00f, 0.36f);
    colors[ImGuiCol_HeaderActive]           = ImVec4(0.20f, 0.22f, 0.23f, 0.33f);
    colors[ImGuiCol_Separator]              = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_SeparatorHovered]       = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_SeparatorActive]        = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_ResizeGrip]             = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_ResizeGripHovered]      = ImVec4(0.44f, 0.44f, 0.44f, 0.29f);
    colors[ImGuiCol_ResizeGripActive]       = ImVec4(0.40f, 0.44f, 0.47f, 1.00f);
    colors[ImGuiCol_Tab]                    = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TabHovered]             = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_TabActive]              = ImVec4(0.20f, 0.20f, 0.20f, 0.36f);
    colors[ImGuiCol_TabUnfocused]           = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TabUnfocusedActive]     = ImVec4(0.14f, 0.14f, 0.14f, 1.00f);
    colors[ImGuiCol_DockingPreview]         = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_DockingEmptyBg]         = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotLines]              = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotLinesHovered]       = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogram]          = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_PlotHistogramHovered]   = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_TableHeaderBg]          = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TableBorderStrong]      = ImVec4(0.00f, 0.00f, 0.00f, 0.52f);
    colors[ImGuiCol_TableBorderLight]       = ImVec4(0.28f, 0.28f, 0.28f, 0.29f);
    colors[ImGuiCol_TableRowBg]             = ImVec4(0.00f, 0.00f, 0.00f, 0.00f);
    colors[ImGuiCol_TableRowBgAlt]          = ImVec4(1.00f, 1.00f, 1.00f, 0.06f);
    colors[ImGuiCol_TextSelectedBg]         = ImVec4(0.20f, 0.22f, 0.23f, 1.00f);
    colors[ImGuiCol_DragDropTarget]         = ImVec4(0.33f, 0.67f, 0.86f, 1.00f);
    colors[ImGuiCol_NavHighlight]           = ImVec4(1.00f, 0.00f, 0.00f, 1.00f);
    colors[ImGuiCol_NavWindowingHighlight]  = ImVec4(1.00f, 0.00f, 0.00f, 0.70f);
    colors[ImGuiCol_NavWindowingDimBg]      = ImVec4(1.00f, 0.00f, 0.00f, 0.20f);
    colors[ImGuiCol_ModalWindowDimBg]       = ImVec4(1.00f, 0.00f, 0.00f, 0.35f);

    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowPadding                     = ImVec2(8.00f, 8.00f);
    style.FramePadding                      = ImVec2(5.00f, 2.00f);
    style.CellPadding                       = ImVec2(6.00f, 6.00f);
    style.ItemSpacing                       = ImVec2(6.00f, 6.00f);
    style.ItemInnerSpacing                  = ImVec2(6.00f, 6.00f);
    style.TouchExtraPadding                 = ImVec2(0.00f, 0.00f);
    style.IndentSpacing                     = 25;
    style.ScrollbarSize                     = 15;
    style.GrabMinSize                       = 10;
    style.WindowBorderSize                  = 1;
    style.ChildBorderSize                   = 1;
    style.PopupBorderSize                   = 1;
    style.FrameBorderSize                   = 1;
    style.TabBorderSize                     = 1;
    style.WindowRounding                    = 7;
    style.ChildRounding                     = 4;
    style.FrameRounding                     = 3;
    style.PopupRounding                     = 4;
    style.ScrollbarRounding                 = 9;
    style.GrabRounding                      = 3;
    style.LogSliderDeadzone                 = 4;
    style.TabRounding                       = 4;

    ImGui_ImplGlfw_InitForVulkan(window->getGLFWWindow(), true);

    _renderContext = RenderContext(_core);
    initFrameResources();

    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = _core->getInstance()->get();
    init_info.PhysicalDevice = _core->getPhysicalDevice();
    init_info.Device = _core->getDevice();
    init_info.QueueFamily = _core->findQueueFamilies(_core->getPhysicalDevice()).graphicsFamily.value();
    init_info.Queue = _core->getGraphicsQueue();
    init_info.PipelineCache = VK_NULL_HANDLE;
    init_info.DescriptorPool = descriptorPool;
    init_info.Subpass = 0;
    init_info.Allocator = VK_NULL_HANDLE;
    init_info.MinImageCount = static_cast<uint32_t>(_core->getSwapchainImageCount());
    init_info.ImageCount = static_cast<uint32_t>(_core->getSwapchainImageCount());
    init_info.CheckVkResultFn = check_vk_result;
    init_info.ColorAttachmentFormat = static_cast<VkFormat>(_core->getSurfaceFormat().format);
    ImGui_ImplVulkan_Init(&init_info, _renderContext.getRenderPass());

    vk::CommandBuffer command_buffer = _core->beginSingleTimeCommands();
    ImGui_ImplVulkan_CreateFontsTexture(command_buffer);
    _core->endSingleTimeCommands(command_buffer);    

    
}

void ImguiRenderPass::initFrameResources(){
    _renderContext.initFramebuffers();
    _renderContext.initCommandBuffers();
}


float drawAverageDensityError(void*, int i) { return simulationMetrics.averageDensityError.get(i) / settings.rho0; };
float drawIterationCount(void*, int i) { return simulationMetrics.iterationCount.get(i); };

void ImguiRenderPass::update(int imageIndex, float dt){
    
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
    
    // ImGui::Begin("Keybindings");
    //     ImGui::BeginTable("Keybindings", 3);
    //         for(auto& keyBinding : gpu::InputManager::getKeyBindings()){
    //             ImGui::TableNextRow();
    //             ImGui::TableSetColumnIndex(0);
    //             ImGui::Text(keyBinding._handle.c_str());
    //             ImGui::TableSetColumnIndex(1);
    //             ImGui::Text(keyBinding._key == 32 ? "Space" : std::string(1, (char)keyBinding._key).c_str());
    //             ImGui::TableSetColumnIndex(2);
    //             static bool editKey = false;
                
    //             if(ImGui::Button("edit")){
    //                 editKey = !editKey;
    //             }
    //             if(editKey){
    //                 gpu::InputManager::suspendKeyInput();
    //                 ImGui::Begin("Press a key", &editKey);
    //                     static char pressedKey[2];
    //                     ImGui::InputText("##pressedKey", pressedKey, IM_ARRAYSIZE(pressedKey));
    //                     if(ImGui::Button("save")){
    //                         gpu::InputManager::updateKeyBinding(keyBinding._handle, (int)toupper(pressedKey[0]));
    //                         editKey = false;
    //                         gpu::InputManager::resumeKeyInput();
    //                     }
    //                 ImGui::End();
    //             }
    //         }
           
    //     ImGui::EndTable();
    // ImGui::End();

    ImGui::Begin("Metrics", &showGPUInfo); 
    {
        ImGui::PlotLines("Average density error", drawAverageDensityError, NULL, SimulationMetrics::MAX_VALUES_PER_METRIC, 0, NULL, 0.0f, settings.maxCompression, ImVec2(0, 80));
        ImGui::PlotLines("IISPH Iteration count", drawIterationCount, NULL, SimulationMetrics::MAX_VALUES_PER_METRIC, 0, NULL, 0, 20, ImVec2(0, 80));
        size_t currentFrame = _core->_swapchainContext._currentFrame;
        if (ImGui::BeginTable("Timings", 2))
        {
            for (int row = 0; row < timestampLabels[currentFrame].size(); row++)
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text(timestampLabels[currentFrame][row].c_str());
                ImGui::TableSetColumnIndex(1);
                ImGui::Text((std::to_string((timestamps[currentFrame][row + 1] - timestamps[currentFrame][row]) / (float)1000000) + " ms").c_str());
            }
            if(!timestampLabels[currentFrame].empty())
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("Total");
                ImGui::TableSetColumnIndex(1);
                ImGui::Text((std::to_string((timestamps[currentFrame][timestampLabels[currentFrame].size() - 1] - timestamps[currentFrame][0]) / (float)1000000) + " ms").c_str());
            }
            {
                ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("dt");
                ImGui::TableSetColumnIndex(1);
                ImGui::Text((std::to_string(settings.dt * 1000.f) + " ms").c_str());
            }
            ImGui::EndTable();
        }
    }
    ImGui::End();

    ImGui::Begin("Scene"); 
    {
        if (ImGui::Button("Wireframe"))
        {
            toggleWireframeCallback();
        }
        if (ImGui::Button("0"))
        {
            changeSceneCallback(0);
        }
        if (ImGui::Button("1"))
        {
            changeSceneCallback(1);
        }
        if (ImGui::Button("2"))
        {
            changeSceneCallback(2);
        }
    }
    ImGui::End();


    ImGui::Begin("Simulation", &showSimulationSettings);
    {
        if (ImGui::Button("Reset"))
        {
            resetSimulation = true;
        }
        if (ImGui::Button(simulationRunning ? "Pause" : "Play"))
        {
           simulationRunning = simulationRunning ? false : true;
        }
        ImGui::BeginTable("", 2);
            ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("Current Frame");
                ImGui::TableSetColumnIndex(1);
                ImGui::Text(std::to_string(currentFrameCount).c_str());
            ImGui::TableNextRow();
                ImGui::TableSetColumnIndex(0);
                ImGui::Text("Simulation speed factor");
                ImGui::TableSetColumnIndex(1);
                ImGui::Text(std::to_string(simulationSpeedFactor).c_str());
        ImGui::EndTable();
        
        ImGui::InputInt("Pause on frame", &pauseOnFrame, -1,10000);

        ImGui::SliderFloat("Rest Density (kg/m^2)", &settings.rho0, 1.f, 3000.f );
        ImGui::InputFloat("Maximum compression", &settings.maxCompression, 0.0001f, 0.001f);
        ImGui::InputFloat("Maximum timestep", &settings.maxTimestep, 0.008f);
        ImGui::SliderFloat("Mass (kg)", &settings.mass, 1.f, 100.f);
        // ImGui::DragFloat("Kernel Radius (m)", &settings.h_LR, 1.f, 0.1f, 100.f);
        // ImGui::DragFloat("Sleeping Speed (m/s)", &settings.sleepingSpeed, 0.05f, 0.01f, 1.f);
        ImGui::SliderAngle("Angle of repose",&settings.theta, 0.f, 90.f, "%.0f°");
        // ImGui::DragFloat("Viscosity constant", &settings.sigma, 1.f, 0.01f, 10.f);
        ImGui::DragFloat3("Gravity (m/s^2)", glm::value_ptr(settings.g));
        ImGui::DragFloat3("Air velocity (m/s)", glm::value_ptr(settings.windDirection));
        ImGui::DragFloat("Air Density", &settings.rhoAir, 1.f, 0.01f, 10.f);
        ImGui::DragFloat("Drag Coefficient", &settings.dragCoefficient, 1.f, 0.01f, 10.f);
    }
    ImGui::End();

    // ImGui::ShowDemoWindow(&show_demo_window);

    ImGui::End();

    ImGui::Render();

    additionalWindows();

    _renderContext.beginCommandBuffer();

    _renderContext.beginRenderPass(imageIndex);

    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), _renderContext.getCommandBuffer());

    _renderContext.endRenderPass();
    _renderContext.endCommandBuffer();
}

void ImguiRenderPass::createDescriptorPool() {
    descriptorPool = _core->createDescriptorPool({
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
    }, 1000);
}

void ImguiRenderPass::destroyFrameResources(){

    _renderContext.destroyFramebuffers();
}

void ImguiRenderPass::destroy(){
    destroyFrameResources();

    _renderContext.freeCommandBuffers();
    _renderContext.destroyRenderPass();
    _core->destroyDescriptorPool(descriptorPool);

    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImguiRenderPass::additionalWindows()
{
    ImGuiIO& io = ImGui::GetIO();
    // Update and Render additional Platform Windows
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
    }
}
