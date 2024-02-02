#version 460

#extension GL_EXT_nonuniform_qualifier : require

layout( push_constant ) uniform Settings{
    vec4 g; 

    float r_LR;         
    float h_LR; 
    float rho0; 
    float mass;

    float maxCompression;	
    float dt;	 
    float DOMAIN_WIDTH; 
    float DOMAIN_HEIGHT;  

    float sleepingSpeed;
    float h_HR;
    float theta;                               
    float rhoAir;                                 
    
    vec4 windDirection;      

    float dragCoefficient;                
    uint n_HR; 
    float pad0;
    float pad1;           
} settings;

layout(location = 0) out vec4 outColor;


layout(set = 1, binding = 0) uniform sampler textureSampler; 
layout(set = 1, binding = 1) uniform texture2D textures[]; 

layout(set = 2, binding = 0) uniform Material{
	vec4 baseColorFactor ;
	vec4 emissiveFactor;
	int baseColorTexture;
	int metallicRoughnessTexture;
	int occlusionTexture;
	int emissiveTexture;
	int specularGlossinessTexture;
	int diffuseTexture;
	float alphaCutoff;
	float metallicFactor;
	float roughnessFactor;
	float emissiveStrength;
	int alphaMode;
} material; 

layout (location = 0) in vec3 vPosition;
layout (location = 1) in vec3 vNormal;
layout (location = 2) in vec2 vUV;

vec3 lightDir = vec3(1.0, 1.0, 1.0);

void main() {
    vec4 baseColor = material.baseColorTexture >= 0 ? texture(sampler2D(textures[material.baseColorTexture], textureSampler), vUV) : vec4(1.0);// vec4(1.0, 0.0, 1.0, 1.0);
    float diff = max(dot(vNormal, lightDir), 0.0);
    outColor = (0.03 + diff) * vec4(material.baseColorFactor * baseColor);
    // outColor = vec4(1.0);
}