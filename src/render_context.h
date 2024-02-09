
#include <vulkan/vulkan.hpp>

namespace gpu
{
    struct RenderSet{
        vk::Framebuffer _framebuffer;
        vk::CommandBuffer _commandBuffer;
    };

    class RenderContext
    {
    private:
        /* data */
        vk::RenderPass _renderPass;
        vk::PipelineLayout _pipelineLayout;
        vk::Pipeline _pipeline;

    public:
        RenderContext(/* args */);
        ~RenderContext();
    };

} // namespace gpu

