#pragma once 
#ifndef HEADER_H
#define HEADER_H
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <string>
#include <glm/glm.hpp>

extern bool simulationRunning;
extern bool simulationStepForward;

extern std::vector<std::string> passTimeings;
#define PARTICLE_RADIUS 0.2f                            //* m
#define PARTICLE_VOLUME (float) M_PI * (PARTICLE_RADIUS * PARTICLE_RADIUS)

struct SPHSettings{
    glm::vec2 g = glm::vec2(0.f, -9.81f);               //* m/s^2
    float particleRadius = PARTICLE_RADIUS;             //* m
    float kernelRadius = particleRadius * 4;            //* m

    float rho0 = 1950.f;                                //* kg/m^3
    float mass = PARTICLE_VOLUME * rho0;                //* kg
    float stiffness = 5000.f;	                        
    float dt = 0.0006f;	                                //* s

    float DOMAIN_WIDTH = 50.f;                          //* m
    float DOMAIN_HEIGHT = 25.f;                         //* m
    float sleepingSpeed = 0.1f;                         //* m/s
    bool upsamplingEnabled = true;

    float theta = 45.f * (float)M_PI / 180.f;           //* rad (angle of repose)
    float sigma = 0.25f;                                //* viscosity coefficient
    float alpha = 0.5f;                                 //* viscosity constant
    float beta = 1.0f;                                  //* cohesion intensity
    
    float C = 10000.0f;                                 //* maximum cohesion
    float dragCoefficient = 0.47f;                      //* Sphere Reynoldsnumber 10^6
    float rhoAir = 1293.f;                              //* Air density
    float pad3;
};

extern SPHSettings settings;

struct UniformBufferObject {
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 proj = glm::mat4(1.0f);
};

#endif