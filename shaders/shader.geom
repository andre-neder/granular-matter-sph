#version 460
layout (points) in;
layout (points, max_vertices = 1) out;
// layout (triangle_strip, max_vertices = 3) out;

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in float[] inRho;

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
    float pad2;

    float theta;       
    float sigma;                           
    float alpha;                             
    float beta;                              
    
    float C;                                
    float dragCoefficient;                
    float rhoAir;                             
    float pad3;
} settings;

vec2 rotate(vec2 x, float deg){
    float rad = radians(deg);
    return mat2(vec2(cos(rad), sin(rad)), vec2(-sin(rad), cos(rad))) * x;
}

vec2 particleShapeOffsets[3] = {
    vec2(0, settings.particleRadius),
    rotate(vec2(0, settings.particleRadius), 120),
    rotate(vec2(0, settings.particleRadius), -120),
};

vec4 transformScreenSpace(vec2 v){
    return vec4((settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT) * (v.x / settings.DOMAIN_WIDTH) * 2.0 - (settings.DOMAIN_WIDTH / settings.DOMAIN_HEIGHT), (v.y / settings.DOMAIN_HEIGHT) * 2.0 - 1.0, 0.0, 1.0);
}

void main() {    
    // gl_Position = gl_in[0].gl_Position;// + vec4(-0.1, 0.0, 0.0, 0.0); 
    // EmitVertex();
    outRho = inRho[0];
    gl_Position =  ubo.proj * ubo.view * ubo.model * transformScreenSpace( gl_in[0].gl_Position.xy) ;// transformScreenSpace(particleShapeOffsets[0]);
    EmitVertex();
    EndPrimitive();
    // outRho = inRho[0];
    // gl_Position =  ubo.proj * ubo.view * ubo.model * transformScreenSpace( gl_in[0].gl_Position.xy + particleShapeOffsets[0]) ;// transformScreenSpace(particleShapeOffsets[0]);
    // EmitVertex();
    // // EndPrimitive();
    // outRho = inRho[0];
    // gl_Position =  ubo.proj * ubo.view * ubo.model * transformScreenSpace( gl_in[0].gl_Position.xy + particleShapeOffsets[1]) ;// transformScreenSpace(particleShapeOffsets[0]);
    // EmitVertex();
    // // EndPrimitive();
    // outRho = inRho[0];
    // gl_Position =  ubo.proj * ubo.view * ubo.model * transformScreenSpace( gl_in[0].gl_Position.xy + particleShapeOffsets[2]) ;// transformScreenSpace(particleShapeOffsets[0]);
    // EmitVertex();
    // EndPrimitive();
} 