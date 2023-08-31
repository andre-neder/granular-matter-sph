#pragma once 
#include "vulkan_utils.h"


PFN_vkCreateDebugUtilsMessengerEXT pfnVkCreateDebugUtilsMessengerEXT;
PFN_vkDestroyDebugUtilsMessengerEXT pfnVkDestroyDebugUtilsMessengerEXT;

bool QueueFamilyIndices::isComplete() {
	return graphicsFamily.has_value() && presentFamily.has_value();
}

#ifdef IMGUI_VULKAN_DEBUG_REPORT
	static VKAPI_ATTR VkBool32 VKAPI_CALL debug_report(VkDebugReportFlagsEXT flags, VkDebugReportObjectTypeEXT objectType, uint64_t object, size_t location, int32_t messageCode, const char* pLayerPrefix, const char* pMessage, void* pUserData)
	{
		(void)flags; (void)object; (void)location; (void)messageCode; (void)pUserData; (void)pLayerPrefix; // Unused arguments
		fprintf(stderr, "[vulkan] Debug report from ObjectType: %i\nMessage: %s\n\n", objectType, pMessage);
		return VK_FALSE;
	}
#endif // IMGUI_VULKAN_DEBUG_REPORT


VKAPI_ATTR VkResult VKAPI_CALL vkCreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT *pCreateInfo, const VkAllocationCallbacks *pAllocator, VkDebugUtilsMessengerEXT *pMessenger){
	return pfnVkCreateDebugUtilsMessengerEXT(instance, pCreateInfo, pAllocator, pMessenger);
}

VKAPI_ATTR void VKAPI_CALL vkDestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT messenger, VkAllocationCallbacks const *pAllocator){
	return pfnVkDestroyDebugUtilsMessengerEXT(instance, messenger, pAllocator);
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageTypes, const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData, void *pUserData)
{
	std::ostringstream message;
	message << vk::to_string(static_cast<vk::DebugUtilsMessageSeverityFlagBitsEXT>(messageSeverity)) << ": "
			<< vk::to_string(static_cast<vk::DebugUtilsMessageTypeFlagsEXT>(messageTypes)) << ":\n";
	message << "\t"
			<< "messageIDName   = <" << pCallbackData->pMessageIdName << ">\n";
	message << "\t"
			<< "messageIdNumber = " << pCallbackData->messageIdNumber << "\n";
	message << "\t"
			<< "message         = <" << pCallbackData->pMessage << ">\n";
	if (0 < pCallbackData->queueLabelCount){
		message << "\t"
				<< "Queue Labels:\n";
		for (uint8_t i = 0; i < pCallbackData->queueLabelCount; i++){
			message << "\t\t"
					<< "labelName = <" << pCallbackData->pQueueLabels[i].pLabelName << ">\n";
		}
	}
	if (0 < pCallbackData->cmdBufLabelCount){
		message << "\t"
				<< "CommandBuffer Labels:\n";
		for (uint8_t i = 0; i < pCallbackData->cmdBufLabelCount; i++){
			message << "\t\t"
					<< "labelName = <" << pCallbackData->pCmdBufLabels[i].pLabelName << ">\n";
		}
	}
	if (0 < pCallbackData->objectCount){
		message << "\t"
				<< "Objects:\n";
		for (uint8_t i = 0; i < pCallbackData->objectCount; i++){
			message << "\t\t"
					<< "Object " << i << "\n";
			message << "\t\t\t"
					<< "objectType   = "
					<< vk::to_string(static_cast<vk::ObjectType>(pCallbackData->pObjects[i].objectType)) << "\n";
			message << "\t\t\t"
					<< "objectHandle = " << pCallbackData->pObjects[i].objectHandle << "\n";
			if (pCallbackData->pObjects[i].pObjectName){
				message << "\t\t\t"
						<< "objectName   = <" << pCallbackData->pObjects[i].pObjectName << ">\n";
			}
		}
	}
	std::cout << message.str() << std::endl;
	return VK_FALSE;
}


vk::DebugUtilsMessageSeverityFlagsEXT messageSeverityFlags = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation;

bool checkValidationLayerSupport() {
	std::vector<vk::LayerProperties> availableLayers = vk::enumerateInstanceLayerProperties();
	for (const char* layerName : validationLayers) {
		bool layerFound = false;
		for (const auto& layerProperties : availableLayers) {
			if (strcmp(layerName, layerProperties.layerName) == 0) {
				layerFound = true;
				break;
			}
		}
		if (!layerFound) {
			return false;
		}
	}
	return true;
}
	
vk::Instance createInstance(bool enableValidation){
	if (enableValidation && !checkValidationLayerSupport()) {
		throw std::runtime_error("validation layers requested, but not available!");
	}
	uint32_t glfwExtensionCount = 0;
	const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
	std::vector<const char*> extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);

	vk::ApplicationInfo applicationInfo("VulkanBase", VK_MAKE_VERSION(0, 0 ,1), "VulkanEngine", 1, VK_API_VERSION_1_1);

	try {
		if(enableValidation){
			extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
			vk::StructureChain<vk::InstanceCreateInfo, vk::DebugUtilsMessengerCreateInfoEXT> instanceCreateInfoChain{
				vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &applicationInfo, validationLayers, extensions),
				vk::DebugUtilsMessengerCreateInfoEXT({}, messageSeverityFlags, messageTypeFlags, debugCallback)
			};
			return vk::createInstance(instanceCreateInfoChain.get<vk::InstanceCreateInfo>());
		}else{
			vk::InstanceCreateInfo instanceCreateInfo(vk::InstanceCreateFlags(), &applicationInfo, {}, extensions);
			return vk::createInstance(instanceCreateInfo);
		}
	}catch(std::exception& e) {
		std::cerr << "Exception Thrown: " << e.what();
	}
}

vk::DebugUtilsMessengerEXT createDebugMessenger(vk::Instance instance) {
	pfnVkCreateDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(instance.getProcAddr("vkCreateDebugUtilsMessengerEXT"));
	if (!pfnVkCreateDebugUtilsMessengerEXT){
		throw std::runtime_error("GetInstanceProcAddr: Unable to find pfnVkCreateDebugUtilsMessengerEXT function.");
	}
	pfnVkDestroyDebugUtilsMessengerEXT = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(instance.getProcAddr("vkDestroyDebugUtilsMessengerEXT"));
	if (!pfnVkDestroyDebugUtilsMessengerEXT){
		throw std::runtime_error("GetInstanceProcAddr: Unable to find pfnVkDestroyDebugUtilsMessengerEXT function.");
	}
	vk::DebugUtilsMessengerCreateInfoEXT debugMessengerCreateInfo({}, messageSeverityFlags, messageTypeFlags, debugCallback);
	try{
		return instance.createDebugUtilsMessengerEXT(debugMessengerCreateInfo, nullptr);
	}catch(std::exception& e) {
		std::cerr << "Exception Thrown: " << e.what();
	}
}

