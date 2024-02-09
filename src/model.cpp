#include <model.h>
#include <iostream>

vk::DescriptorSetLayout Model::texturesLayout = VK_NULL_HANDLE;
vk::DescriptorSetLayout Model::materialLayout = VK_NULL_HANDLE;

VertexInputDescription Vertex::get_vertex_description()
{
	VertexInputDescription description;

	// we will have just 1 vertex buffer binding, with a per-vertex rate
	vk::VertexInputBindingDescription mainBinding = {};
	mainBinding.binding = 0;
	mainBinding.stride = sizeof(Vertex);
	mainBinding.inputRate = vk::VertexInputRate::eVertex;
	description.bindings.push_back(mainBinding);

	// Position will be stored at Location 0
	vk::VertexInputAttributeDescription posAttribute = {};
	posAttribute.binding = 0;
	posAttribute.location = 0;
	posAttribute.format = vk::Format::eR32G32B32Sfloat;
	posAttribute.offset = offsetof(Vertex, pos);

	// Normal will be stored at Location 1
	vk::VertexInputAttributeDescription normalAttribute = {};
	normalAttribute.binding = 0;
	normalAttribute.location = 1;
	normalAttribute.format = vk::Format::eR32G32B32Sfloat;
	normalAttribute.offset = offsetof(Vertex, normal);

	// Position will be stored at Location 2
	vk::VertexInputAttributeDescription uvAttribute = {};
	uvAttribute.binding = 0;
	uvAttribute.location = 2;
	uvAttribute.format = vk::Format::eR32G32Sfloat;
	uvAttribute.offset = offsetof(Vertex, uv);

	// Position will be stored at Location 3
	vk::VertexInputAttributeDescription colorAttribute = {};
	colorAttribute.binding = 0;
	colorAttribute.location = 3;
	colorAttribute.format = vk::Format::eR32G32B32A32Sfloat;
	colorAttribute.offset = offsetof(Vertex, color);

	// Position will be stored at Location 4
	vk::VertexInputAttributeDescription jointAttribute = {};
	jointAttribute.binding = 0;
	jointAttribute.location = 4;
	jointAttribute.format = vk::Format::eR32G32B32A32Sfloat;
	jointAttribute.offset = offsetof(Vertex, joint0);

	// Position will be stored at Location 5
	vk::VertexInputAttributeDescription weightAttribute = {};
	weightAttribute.binding = 0;
	weightAttribute.location = 5;
	weightAttribute.format = vk::Format::eR32G32B32A32Sfloat;
	weightAttribute.offset = offsetof(Vertex, weight0);

	// Position will be stored at Location 6
	vk::VertexInputAttributeDescription tangentAttribute = {};
	tangentAttribute.binding = 0;
	tangentAttribute.location = 6;
	tangentAttribute.format = vk::Format::eR32G32B32A32Sfloat;
	tangentAttribute.offset = offsetof(Vertex, tangent);

	description.attributes.push_back(posAttribute);
	description.attributes.push_back(normalAttribute);
	description.attributes.push_back(uvAttribute);
	description.attributes.push_back(colorAttribute);
	description.attributes.push_back(jointAttribute);
	description.attributes.push_back(weightAttribute);
	description.attributes.push_back(tangentAttribute);
	return description;
}

vk::DescriptorSetLayout Model::getTexturesLayout(gpu::Core *core)
{
	if(Model::texturesLayout == VK_NULL_HANDLE){
		Model::texturesLayout = core->createDescriptorSetLayout({
			{0, vk::DescriptorType::eSampler, vk::ShaderStageFlagBits::eFragment},
			{1, vk::DescriptorType::eSampledImage, 7, vk::ShaderStageFlagBits::eFragment, vk::DescriptorBindingFlagBits::eVariableDescriptorCount | vk::DescriptorBindingFlagBits::ePartiallyBound }
		});
	}
	
	return Model::texturesLayout;
}

vk::DescriptorSetLayout Model::getMaterialLayout(gpu::Core *core)
{
    if(Model::materialLayout == VK_NULL_HANDLE){
		Model::materialLayout = core->createDescriptorSetLayout({
        	{	0, vk::DescriptorType::eUniformBuffer, vk::ShaderStageFlagBits::eFragment}
		});
	}
	return Model::materialLayout;
}

void Model::cleanupDescriptorSetLayouts(gpu::Core *core)
{
	if(Model::texturesLayout != VK_NULL_HANDLE){
		core->destroyDescriptorSetLayout(Model::texturesLayout);
	}
	if(Model::materialLayout != VK_NULL_HANDLE){
		core->destroyDescriptorSetLayout(Model::materialLayout);
	}
}

Model::Model() : core() {}
Model::Model(gpu::Core* core): core(core){}

void Model::createBuffers()
{
	
	indexBuffer = core->bufferFromData(_indices.data(), _indices.size() * sizeof(uint32_t), vk::BufferUsageFlagBits::eIndexBuffer, vma::MemoryUsage::eAutoPreferDevice);
	vertexBuffer = core->bufferFromData(_vertices.data(), _vertices.size() * sizeof(Vertex), vk::BufferUsageFlagBits::eVertexBuffer, vma::MemoryUsage::eAutoPreferDevice);
}


void Model::destroy()
{
	if(core != nullptr){
		core->destroySampler(textureSampler);
		core->destroyDescriptorPool(descriptorPool);
		for(auto&& image : images){
			core->destroyImage(image);
		}
		for(auto&& view : views){
			core->destroyImageView(view);
		}
		int index = 0;
		core->destroyDescriptorPool(materialDescriptorPool);
		for(auto material : _materials){

			core->destroyBuffer(materialBuffers[index]);
			index++;
		}

		core->destroyBuffer(indexBuffer);
		core->destroyBuffer(vertexBuffer);
	}
}

void Model::loadImages(tinygltf::Model &input)
{

	for (tinygltf::Image &image : input.images) {
		unsigned char* buffer = nullptr;
		vk::DeviceSize bufferSize = 0;
		
		bool deleteBuffer = false;
		if (image.component == 3) {
			bufferSize = image.width * image.height * 4;
			buffer = new unsigned char[bufferSize];
			unsigned char* rgba = buffer;
			unsigned char* rgb = &image.image[0];
			for (size_t i = 0; i < image.width * image.height; ++i) {
				for (int32_t j = 0; j < 3; ++j) {
					rgba[j] = rgb[j];
				}
				rgba += 4;
				rgb += 3;
			}
			deleteBuffer = true;
		}
		else {
			buffer = &image.image[0];
			bufferSize = image.image.size();
		}

		vk::Format format = vk::Format::eR8G8B8A8Unorm;
		uint32_t width = image.width;
		uint32_t height = image.height;

		auto formatProperties = core->getPhysicalDevice().getFormatProperties(format);
		if(!(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitSrc) || !(formatProperties.optimalTilingFeatures & vk::FormatFeatureFlagBits::eBlitDst))
		{
			throw std::runtime_error("unsported Image Format!");
		}

        auto image = core->image2DFromData(buffer, vk::ImageUsageFlagBits::eSampled, vma::MemoryUsage::eAutoPreferDevice, {}, 1, 1, vk::Format::eR8G8B8A8Unorm);
		images.push_back(image);

		if (deleteBuffer) {
            delete[] buffer;
        }
	}

	unsigned char* buffer = new unsigned char[4];
	memset(buffer, 0, 4);

	
    auto image = core->image2DFromData(buffer, vk::ImageUsageFlagBits::eSampled, vma::MemoryUsage::eAutoPreferDevice, {}, 1, 1, vk::Format::eR8G8B8A8Unorm);
	images.push_back(image);
}

void Model::loadMaterials(tinygltf::Model &input)
{
	uint32_t texture_count = static_cast<uint32_t>(images.size() - 1);
	for (tinygltf::Material &mat : input.materials)
	{

		Material material;

		if (mat.values.find("roughnessFactor") != mat.values.end())
		{
			material.roughnessFactor = static_cast<float>(mat.values["roughnessFactor"].Factor());
		}
		if (mat.values.find("metallicFactor") != mat.values.end())
		{
			material.metallicFactor = static_cast<float>(mat.values["metallicFactor"].Factor());
		}
		if (mat.values.find("emissiveFactor") != mat.values.end())
		{
			material.emissiveFactor = glm::vec4(glm::make_vec3(mat.values["emissiveFactor"].ColorFactor().data()), 1.0f);
		}
		if (mat.extensions.find("KHR_materials_emissive_strength") != mat.extensions.end())
		{
			auto ext = mat.extensions.find("KHR_materials_emissive_strength");
			if (ext->second.Has("emissiveStrength")) {
				auto value = ext->second.Get("emissiveStrength");
				material.emissiveStrength = static_cast<float>(value.Get<double>());
			}
		}
		if (mat.values.find("baseColorFactor") != mat.values.end())
		{
			material.baseColorFactor = glm::make_vec4(mat.values["baseColorFactor"].ColorFactor().data());
		}
		if (mat.additionalValues.find("alphaMode") != mat.additionalValues.end())
		{
			tinygltf::Parameter param = mat.additionalValues["alphaMode"];
			if (param.string_value == "BLEND")
			{
				material.alphaMode = Material::ALPHAMODE_BLEND;
			}
			if (param.string_value == "MASK")
			{
				material.alphaMode = Material::ALPHAMODE_MASK;
			}
		}
		if (mat.additionalValues.find("alphaCutoff") != mat.additionalValues.end())
		{
			material.alphaCutoff = static_cast<float>(mat.additionalValues["alphaCutoff"].Factor());
		}
		if (mat.values.find("baseColorTexture") != mat.values.end()) {
			material.baseColorTexture = getTextureIndex(input.textures[mat.values["baseColorTexture"].TextureIndex()].source);
		}
		if (mat.values.find("metallicRoughnessTexture") != mat.values.end()) {
			int32_t metallicRoughnessTextureIndex = getTextureIndex(input.textures[mat.values["metallicRoughnessTexture"].TextureIndex()].source);
		}
		if (mat.values.find("normalTexture") != mat.values.end()) {
			int32_t normalTextureIndex = getTextureIndex(input.textures[mat.values["normalTexture"].TextureIndex()].source);
		}
		if (mat.values.find("emissiveTexture") != mat.values.end()) {
			int32_t emissiveTextureIndex = getTextureIndex(input.textures[mat.values["emissiveTexture"].TextureIndex()].source);
		}
		if (mat.values.find("occlusionTexture") != mat.values.end()) {
			int32_t occlusionTextureIndex = getTextureIndex(input.textures[mat.values["occlusionTexture"].TextureIndex()].source);
		}
		if (mat.values.find("specularGlossinessTexture") != mat.values.end()) {
			int32_t specularGlossinessTextureIndex = getTextureIndex(input.textures[mat.values["specularGlossinessTexture"].TextureIndex()].source);
		}
		if (mat.values.find("diffuseTexture") != mat.values.end()) {
			int32_t diffuseTextureIndex = getTextureIndex(input.textures[mat.values["diffuseTexture"].TextureIndex()].source);
		}
		_materials.push_back(material);
	}
	_materials.push_back(Material());
}

uint32_t Model::getTextureIndex(uint32_t index)
{
	if (index < images.size() && index >= 0) {
		return index;
	}
	return static_cast<uint32_t>(images.size() - 1);
}

void Model::loadNode(const tinygltf::Node &inputNode, const tinygltf::Model &input, Node *parent, std::vector<uint32_t> &indexBuffer, std::vector<Vertex> &vertexBuffer)
{
	Node *node = new Node{};
	node->name = inputNode.name;
	node->parent = parent;

	// Get the local node matrix
	// It's either made up from translation, rotation, scale or a 4x4 matrix
	node->matrix = glm::mat4(1.0f);
	if (inputNode.translation.size() == 3)
	{
		node->matrix = glm::translate(node->matrix, glm::vec3(glm::make_vec3(inputNode.translation.data())));
	}
	if (inputNode.rotation.size() == 4)
	{
		glm::quat q = glm::make_quat(inputNode.rotation.data());
		node->matrix *= glm::mat4(q);
	}
	if (inputNode.scale.size() == 3)
	{
		node->matrix = glm::scale(node->matrix, glm::vec3(glm::make_vec3(inputNode.scale.data())));
	}
	if (inputNode.matrix.size() == 16)
	{
		node->matrix = glm::make_mat4x4(inputNode.matrix.data());
	};

	// Load node's children
	if (inputNode.children.size() > 0)
	{
		for (size_t i = 0; i < inputNode.children.size(); i++)
		{
			loadNode(input.nodes[inputNode.children[i]], input, node, indexBuffer, vertexBuffer);
		}
	}

	// If the node contains mesh data, we load vertices and indices from the buffers
	// In glTF this is done via accessors and buffer views
	if (inputNode.mesh > -1)
	{
		const tinygltf::Mesh mesh = input.meshes[inputNode.mesh];
		// Iterate through all primitives of this node's mesh
		for (size_t i = 0; i < mesh.primitives.size(); i++)
		{
			const tinygltf::Primitive &glTFPrimitive = mesh.primitives[i];
			uint32_t firstIndex = static_cast<uint32_t>(indexBuffer.size());
			uint32_t firstVertex = static_cast<uint32_t>(vertexBuffer.size());
			uint32_t indexCount = 0;
			uint32_t vertexCount = 0;
			// Vertices
			{
				const float *positionBuffer = nullptr;
				const float *normalsBuffer = nullptr;
				const float *texCoordsBuffer = nullptr;
				const float *tangentsBuffer = nullptr;

				// Get buffer data for vertex normals
				if (glTFPrimitive.attributes.find("POSITION") != glTFPrimitive.attributes.end())
				{
					const tinygltf::Accessor &accessor = input.accessors[glTFPrimitive.attributes.find("POSITION")->second];
					const tinygltf::BufferView &view = input.bufferViews[accessor.bufferView];
					positionBuffer = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
					vertexCount = static_cast<uint32_t>(accessor.count);
				}
				// Get buffer data for vertex normals
				if (glTFPrimitive.attributes.find("NORMAL") != glTFPrimitive.attributes.end())
				{
					const tinygltf::Accessor &accessor = input.accessors[glTFPrimitive.attributes.find("NORMAL")->second];
					const tinygltf::BufferView &view = input.bufferViews[accessor.bufferView];
					normalsBuffer = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
				}
				// Get buffer data for vertex texture coordinates
				// glTF supports multiple sets, we only load the first one
				if (glTFPrimitive.attributes.find("TEXCOORD_0") != glTFPrimitive.attributes.end())
				{
					const tinygltf::Accessor &accessor = input.accessors[glTFPrimitive.attributes.find("TEXCOORD_0")->second];
					const tinygltf::BufferView &view = input.bufferViews[accessor.bufferView];
					texCoordsBuffer = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
				}
				// POI: This sample uses normal mapping, so we also need to load the tangents from the glTF file
				if (glTFPrimitive.attributes.find("TANGENT") != glTFPrimitive.attributes.end())
				{
					const tinygltf::Accessor &accessor = input.accessors[glTFPrimitive.attributes.find("TANGENT")->second];
					const tinygltf::BufferView &view = input.bufferViews[accessor.bufferView];
					tangentsBuffer = reinterpret_cast<const float *>(&(input.buffers[view.buffer].data[accessor.byteOffset + view.byteOffset]));
				}

				// Append data to model's vertex buffer
				for (size_t v = 0; v < vertexCount; v++)
				{
					Vertex vert{};
					vert.pos = glm::vec4(glm::make_vec3(&positionBuffer[v * 3]), 1.0f);
					vert.normal = glm::normalize(glm::vec3(normalsBuffer ? glm::make_vec3(&normalsBuffer[v * 3]) : glm::vec3(0.0f)));
					vert.uv = texCoordsBuffer ? glm::make_vec2(&texCoordsBuffer[v * 2]) : glm::vec3(0.0f);
					vert.color = glm::vec4(1.0f);
					vert.tangent = tangentsBuffer ? glm::make_vec4(&tangentsBuffer[v * 4]) : glm::vec4(0.0f);
					vertexBuffer.push_back(vert);
				}
			}
			// Indices
			{
				const tinygltf::Accessor &accessor = input.accessors[glTFPrimitive.indices];
				const tinygltf::BufferView &bufferView = input.bufferViews[accessor.bufferView];
				const tinygltf::Buffer &buffer = input.buffers[bufferView.buffer];

				indexCount += static_cast<uint32_t>(accessor.count);

				// glTF supports different component types of indices
				switch (accessor.componentType)
				{
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_INT:
				{
					const uint32_t *buf = reinterpret_cast<const uint32_t *>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
					for (size_t index = 0; index < accessor.count; index++)
					{
						indexBuffer.push_back(buf[index] + firstVertex);
					}
					break;
				}
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_SHORT:
				{
					const uint16_t *buf = reinterpret_cast<const uint16_t *>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
					for (size_t index = 0; index < accessor.count; index++)
					{
						indexBuffer.push_back(buf[index] + firstVertex);
					}
					break;
				}
				case TINYGLTF_PARAMETER_TYPE_UNSIGNED_BYTE:
				{
					const uint8_t *buf = reinterpret_cast<const uint8_t *>(&buffer.data[accessor.byteOffset + bufferView.byteOffset]);
					for (size_t index = 0; index < accessor.count; index++)
					{
						indexBuffer.push_back(buf[index] + firstVertex);
					}
					break;
				}
				default:
					std::cerr << "Index component type " << accessor.componentType << " not supported!" << std::endl;
					return;
				}
			}
			uint32_t materialIndex = glTFPrimitive.material > -1 ? glTFPrimitive.material : _materials.size() - 1;
			Primitive *primitive = new Primitive(firstIndex, indexCount, firstVertex, vertexCount, materialIndex);
			node->primitives.push_back(primitive);
		}
	}

	if (parent)
	{
		parent->children.push_back(node);
	}
	else
	{
		_nodes.push_back(node);
	}
	_linearNodes.push_back(node);
}

bool Model::load_from_glb(const char *filename)
{
	tinygltf::Model glTFInput;
	tinygltf::TinyGLTF gltfContext;
	std::string error, warning;

	bool fileLoaded = gltfContext.LoadBinaryFromFile(&glTFInput, &error, &warning, filename);

	if (fileLoaded)
	{
		if(core != nullptr){
			loadImages(glTFInput);
		}
		loadMaterials(glTFInput);
		const tinygltf::Scene &scene = glTFInput.scenes[0];
		for (size_t i = 0; i < scene.nodes.size(); i++)
		{
			const tinygltf::Node node = glTFInput.nodes[scene.nodes[i]];
			loadNode(node, glTFInput, nullptr, _indices, _vertices);
		}

		for (auto node : _linearNodes)
		{
			if (node->primitives.size() > 0)
			{
				for (auto primitive : node->primitives)
				{
					if (primitive->indexCount > 0)
					{
						vk::TransformMatrixKHR transformMatrix{};
						auto m = glm::mat3x4(glm::transpose(node->getMatrix()));
						memcpy(&transformMatrix, (void *)&m, sizeof(glm::mat3x4));
						_transforms.push_back(transformMatrix);
					}
				}
			}
		}
		
		if(core != nullptr){
			createBuffers();
			createDescriptorSet();
		}
	}
	else
	{
		return false;
	}
	return true;
}


void Model::createDescriptorSet()
{

	// create sampler
	textureSampler = core->createSampler(vk::SamplerAddressMode::eRepeat);

	// create texture views
	for(auto image : images){
		auto view = core->createImageView2D(image, vk::Format::eR8G8B8A8Unorm); 
        views.push_back(view);
	}
	
	// create descriptor set for all textures
	descriptorPool = core->createDescriptorPool({
        { vk::DescriptorType::eSampler, 1 * gpu::MAX_FRAMES_IN_FLIGHT },
        { vk::DescriptorType::eSampledImage, static_cast<uint32_t>(views.size())  * gpu::MAX_FRAMES_IN_FLIGHT },
    }, static_cast<uint32_t>(views.size() + 1) * gpu::MAX_FRAMES_IN_FLIGHT);

	descriptorSets = core->allocateDescriptorSets(Model::getTexturesLayout(core), descriptorPool, gpu::MAX_FRAMES_IN_FLIGHT);


	for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
        core->addDescriptorWrite(descriptorSets[i], 
			{ 0, vk::DescriptorType::eSampler, textureSampler, {}, {} }
		);
		core->addDescriptorWrite(descriptorSets[i], 
			{ 1, vk::DescriptorType::eSampledImage, {}, views, vk::ImageLayout::eShaderReadOnlyOptimal }
		);

        core->updateDescriptorSet(descriptorSets[i]);
	}
	// create descriptorset for each material
	materialDescriptorPool = core->createDescriptorPool({
			{ vk::DescriptorType::eUniformBuffer, static_cast<uint32_t>(1 * _materials.size()) * gpu::MAX_FRAMES_IN_FLIGHT }, 
		}, (2 * _materials.size()) * gpu::MAX_FRAMES_IN_FLIGHT);
	
	int index = 0;
	materialDescriptorSets.resize(_materials.size());
	materialBuffers.resize(_materials.size());
	for(auto material : _materials){

		materialDescriptorSets[index] = core->allocateDescriptorSets(Model::getMaterialLayout(core), materialDescriptorPool, gpu::MAX_FRAMES_IN_FLIGHT);
		materialBuffers[index] = core->bufferFromData(&material, sizeof(Material), vk::BufferUsageFlagBits::eUniformBuffer, vma::MemoryUsage::eAutoPreferDevice);
		
		for (size_t i = 0; i < gpu::MAX_FRAMES_IN_FLIGHT; i++) {
			core->addDescriptorWrite(materialDescriptorSets[index][i], 
				{ 0, vk::DescriptorType::eUniformBuffer, materialBuffers[index], sizeof(Material) }
			);
			core->updateDescriptorSet(materialDescriptorSets[index][i]);
		}
		index++;
	}
}

glm::mat4 Node::localMatrix()
{
	return glm::translate(glm::mat4(1.0f), translation) * glm::mat4(rotation) * glm::scale(glm::mat4(1.0f), scale) * matrix;
}

glm::mat4 Node::getMatrix()
{
	glm::mat4 m = localMatrix();
	Node *p = parent;
	while (p)
	{
		m = p->localMatrix() * m;
		p = p->parent;
	}
	return m;
}