cmake_minimum_required(VERSION 3.7)
project("VkBase" VERSION 1.0.0)

add_compile_definitions(
    SHADER_PATH="${PROJECT_SOURCE_DIR}/shaders" 
    ASSETS_PATH="${PROJECT_SOURCE_DIR}/assets" )

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
IF(APPLE)
    add_definitions(-Wno-nullability-completeness)
    add_definitions(-Wno-deprecated)
ENDIF()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

IF(MSVC)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
ENDIF()

file(GLOB SOURCES
    src/*.h
    src/*.hpp
    src/*.cpp
    src/vendor/*.h
    src/vendor/*.hpp
    src/vendor/*.cpp
)

add_executable(${CMAKE_PROJECT_NAME} ${SOURCES})

#add vulkan
find_package(Vulkan REQUIRED)
IF (Vulkan_FOUND)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${Vulkan_LIBRARIES})
    target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${Vulkan_INCLUDE_DIR})
ELSE()
    message(ERROR "Vulkan SDK has to be installed")
ENDIF()


#add glfw
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/glfw)

#add glm
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/glm)

#add vma
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/VulkanMemoryAllocator)

#add tinygltf & stb_image
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/tinygltf)

#add glslang & spirv
set(ENABLE_OPT OFF)
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/glslang)

#add imgui
set(USE_BACKEND_VULKAN ON)
set(USE_BACKEND_GLFW ON)
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/imgui)

#add libraries
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE 
    glslang 
    SPIRV 
    glm 
    glfw 
    GPUOpen::VulkanMemoryAllocator 
    tinygltf
    imgui 
)

#add include dirs
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE 
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/vendor/vma-hpp
    ${PROJECT_SOURCE_DIR}/vendor/tinygltf
    ${PROJECT_SOURCE_DIR}/vendor/imgui
    ${PROJECT_SOURCE_DIR}/vendor/TriangleMeshDistance/TriangleMeshDistance/include
)