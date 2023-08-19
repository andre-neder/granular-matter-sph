#include <vulkan/vulkan.hpp>
#include <glslang/SPIRV/GlslangToSpv.h>
#include <iostream>
#include <sstream>
#include <fstream>

class SpirvHelper
{
private: 
	static void InitResources(TBuiltInResource &Resources);
	static EShLanguage FindLanguage(const vk::ShaderStageFlagBits shader_type);
public:
	static void Init();
	static void Finalize();
	static bool GLSLtoSPV(const vk::ShaderStageFlagBits shader_type, const std::string &filename, std::vector<unsigned int> &spirv);
};