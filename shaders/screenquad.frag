#version 460

// layout(binding = 1) uniform sampler3D texSampler; 
// texture(texSampler, fragTexCoord).rgb

// layout(location = 0) in vec3 fragColor;
// layout(location = 1) in vec3 fragTexCoord;

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

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

// float sdBox( vec3 p, vec3 b )
// {
//   return sdBox(vec3(p, 0), vec3(b, 0));
// }

layout(location = 0) out vec4 outColor;
layout(location = 1) in vec3 inPos;

void main() {
    outColor = vec4(246.f / 255.f,215.f / 255.f,176.f / 255.f, 1.0);

    // float sdf = sdBox(inPos, vec3(0.5, 0.5));

    // outColor = vec4(vec3(sdf, 0, 1), 1);
}