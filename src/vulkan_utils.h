#pragma once 
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <iostream>
#include <sstream>
#include <vector>
#include <optional>
#include <set>

struct QueueFamilyIndices {
	std::optional<uint32_t> graphicsFamily;
	std::optional<uint32_t> presentFamily;
	std::optional<uint32_t> computeFamily;
	bool isComplete();
};

struct SwapChainSupportDetails {
	vk::SurfaceCapabilitiesKHR capabilities;
	std::vector<vk::SurfaceFormatKHR> formats;
	std::vector<vk::PresentModeKHR> presentModes;
};

const std::vector<const char*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

bool checkValidationLayerSupport();
	
vk::Instance createInstance(bool enableValidation);

vk::DebugUtilsMessengerEXT createDebugMessenger(vk::Instance instance);
