#pragma once
#include <vulkan/vulkan.hpp>

namespace gpu
{
    namespace Initializers{
        static vk::PipelineMultisampleStateCreateInfo DefaultMultisampleStateCreateInfo({}, vk::SampleCountFlagBits::e1, VK_FALSE);
        static vk::PipelineColorBlendAttachmentState DefaultColorBlendAttachmentState(VK_FALSE, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd, vk::BlendFactor::eOne, vk::BlendFactor::eZero, vk::BlendOp::eAdd,  vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA);
        static vk::PipelineColorBlendStateCreateInfo DefaultColorBlendStateCreateInfo({}, VK_FALSE, vk::LogicOp::eCopy, DefaultColorBlendAttachmentState);
        static vk::PipelineDepthStencilStateCreateInfo DefaultDepthStenilStateCreateInfo({}, VK_TRUE, VK_TRUE, vk::CompareOp::eLess, VK_FALSE, VK_FALSE);

        static std::vector<vk::DynamicState> ViewportScissorDynamicState = {
            vk::DynamicState::eViewport,
            vk::DynamicState::eScissor
        };
        static vk::PipelineDynamicStateCreateInfo DefaultDynamicStateCreateInfo({}, static_cast<uint32_t>(ViewportScissorDynamicState.size()), ViewportScissorDynamicState.data());
        static vk::PipelineViewportStateCreateInfo DynamicViewportState({}, 1, nullptr, 1, nullptr);
    };
    
} // namespace gpu
