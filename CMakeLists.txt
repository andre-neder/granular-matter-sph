cmake_minimum_required(VERSION 3.7)
project("Vulkan_KHR_Dynamic_Rendering" VERSION 1.0.0)

add_compile_definitions(
    SHADER_PATH="${PROJECT_SOURCE_DIR}/shaders" 
    ASSETS_PATH="${PROJECT_SOURCE_DIR}/assets" )

add_definitions(-D_CRT_SECURE_NO_WARNINGS)
IF(APPLE)
    add_definitions(-Wno-nullability-completeness)
    add_definitions(-Wno-deprecated)
ENDIF()

set(CMAKE_CXX_STANDARD 17)
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
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE glfw)

#add glm
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/glm)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE glm)

#add vma
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/VulkanMemoryAllocator)
# find_package(VulkanMemoryAllocator CONFIG REQUIRED)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE VulkanMemoryAllocator)

#add tinygltf & stb_image
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/tinygltf)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE tinygltf)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/vendor/tinygltf)

#add glslang
set(ENABLE_OPT OFF)
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/glslang)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE glslang SPIRV)

#add imgui
set(USE_BACKEND_VULKAN ON)
set(USE_BACKEND_GLFW ON)
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/imgui)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE vendor/imgui)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE imgui)

#add discregrid
set(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS ON)
include_directories(${PROJECT_SOURCE_DIR}/vendor/Discregrid/extern)
add_subdirectory(${PROJECT_SOURCE_DIR}/vendor/Discregrid/discregrid)
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE Discregrid)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE vendor/Discregrid/discregrid/include)