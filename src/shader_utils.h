#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <shaderc/shaderc.hpp>

std::string preprocess_shader(const std::string& source_name, shaderc_shader_kind kind, const std::string& source);

std::string compile_file_to_assembly(const std::string& source_name, shaderc_shader_kind kind, const std::string& source, bool optimize = false);

std::vector<uint32_t> compile_file(const std::string& source_name, shaderc_shader_kind kind,  const std::string& source, bool optimize = false);