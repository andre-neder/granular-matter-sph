#pragma once
#include <core.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <stb_image.h>
#include <stb_image_write.h>
#include <tiny_gltf.h>

struct VertexInputDescription
{
	std::vector<vk::VertexInputBindingDescription> bindings;
	std::vector<vk::VertexInputAttributeDescription> attributes;
	vk::PipelineVertexInputStateCreateFlags flags;
};

struct Texture
{
	// vkutils::AllocatedImage image;
    vk::Image image;
	uint32_t index;
	vk::DescriptorImageInfo descriptor;
};

struct Material
{
	enum AlphaMode
	{
		ALPHAMODE_OPAQUE,
		ALPHAMODE_MASK,
		ALPHAMODE_BLEND
	};
	glm::vec4 baseColorFactor = glm::vec4(1.0f);
	glm::vec4 emissiveFactor = glm::vec4(1.0f);
	int32_t baseColorTexture = -1;
	int32_t metallicRoughnessTexture = -1;
	int32_t normalTexture = -1;
	int32_t occlusionTexture = -1;
	int32_t emissiveTexture = -1;
	int32_t specularGlossinessTexture = -1;
	int32_t diffuseTexture = -1;
	float alphaCutoff = 1.0f;
	float metallicFactor = 1.0f;
	float roughnessFactor = 1.0f;
	float emissiveStrength = 0.0f;
	AlphaMode alphaMode = ALPHAMODE_OPAQUE;
};

struct Primitive
{
	uint32_t firstIndex;
	uint32_t indexCount;
	uint32_t firstVertex;
	uint32_t vertexCount;
	Material& material;

	Primitive(uint32_t firstIndex, uint32_t indexCount, uint32_t firstVertex, uint32_t vertexCount, Material& material) : firstIndex(firstIndex), indexCount(indexCount), firstVertex(firstVertex), vertexCount(vertexCount), material(material) {};
};

struct Node
{
	Node *parent;
	std::vector<Node *> children;
	glm::mat4 matrix;
	std::string name;
	std::vector<Primitive *> primitives;
	glm::vec3 translation{};
	glm::vec3 scale{1.0f};
	glm::quat rotation{};
	glm::mat4 localMatrix();
	glm::mat4 getMatrix();
	~Node();
};

struct Vertex
{
	glm::vec3 pos;
	float pad0;
	glm::vec3 normal;
	float pad1;
	glm::vec2 uv;
	float pad2[2];
	glm::vec4 color;
	glm::vec4 joint0;
	glm::vec4 weight0;
	glm::vec4 tangent;
	static VertexInputDescription get_vertex_description();
};

class Model
{
public:
	std::vector<Node *> _nodes{};
	std::vector<Node *> _linearNodes{};
	std::vector<uint32_t> _indices{};
	std::vector<Vertex> _vertices{};
    
	std::vector<Texture> _textures{};
	std::vector<Material> _materials{};
	std::vector<vk::TransformMatrixKHR> _transforms{};
	vk::Sampler _sampler;
	gpu::Core* core;
	Model();
	void createBuffers(gpu::Core* core);
	void destroyBuffers(gpu::Core* core);
	void destroy();
	bool load_from_glb(const char *filename);
	// std::vector<vk::DescriptorImageInfo> getTextureDescriptors();
	vk::Buffer vertexBuffer;
	vk::Buffer indexBuffer;
private:
	void loadImages(tinygltf::Model &input);
	void loadMaterials(tinygltf::Model &input);
	void loadNode(const tinygltf::Node &inputNode, const tinygltf::Model &input, Node *parent, std::vector<uint32_t> &indexBuffer, std::vector<Vertex> &vertexBuffer);
	void loadMaterials(const tinygltf::Model &input);
	uint32_t getTextureIndex(uint32_t index);

	
};