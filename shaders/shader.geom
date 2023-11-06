#version 460
layout (points) in;
layout (points, max_vertices = 5) out;
// layout (triangle_strip, max_vertices = 3) out;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in float[] inRho;
layout(location = 1) in vec2[] inPosition;

layout(location = 0) out float outRho;

layout( push_constant ) uniform Settings{
    vec2 g; 
    float particleRadius;                 
    float kernelRadius; 

    float rho0; 
    float mass;
    float stiffness;	  
    float dt;	 

    float DOMAIN_WIDTH; 
    float DOMAIN_HEIGHT;  
    float sleepingSpeed;
    bool upsamplingEnabled;

    float theta;       
    float sigma;                           
    float alpha;                             
    float beta;                              
    
    float C;                                
    float dragCoefficient;                
    float rhoAir;                             
    float pad3;
} settings;

vec4 transformScreenSpace(vec2 v){
    return ubo.proj * ubo.view * ubo.model * vec4((settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) * (v.x / settings.DOMAIN_WIDTH) * 2.0 - (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT), (v.y / settings.DOMAIN_HEIGHT) * 2.0 - 1.0, 0.0, 1.0);
}

void main() {    
    outRho = inRho[0];
    gl_Position = gl_in[0].gl_Position;
    EmitVertex();
    EndPrimitive();
    if(settings.upsamplingEnabled){
        gl_Position = transformScreenSpace(inPosition[0] + vec2(0, 1) * settings.particleRadius);
        EmitVertex();
        EndPrimitive();

        gl_Position = transformScreenSpace(inPosition[0] + vec2(0, -1) * settings.particleRadius);
        EmitVertex();
        EndPrimitive();

        gl_Position = transformScreenSpace(inPosition[0] + vec2(1, 0) * settings.particleRadius);
        EmitVertex();
        EndPrimitive();

        gl_Position = transformScreenSpace(inPosition[0] + vec2(-1, 0) * settings.particleRadius);
        EmitVertex();
        EndPrimitive();
    }
} 