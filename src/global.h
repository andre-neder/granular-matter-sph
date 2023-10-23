#pragma once 
#ifndef HEADER_H
#define HEADER_H
#define _USE_MATH_DEFINES
#include <math.h>
#include <glm/glm.hpp>

extern bool simulationRunning;
extern bool simulationStepForward;

#define volume (float) M_PI * (particleRadius * particleRadius)

struct SPHSettings{
    glm::vec2 G = glm::vec2(0.f, -9.81f); //* m/s^2
    float particleRadius = 0.1f;                     //* m
    float kernelRadius = particleRadius * 4; 

    float rho0 = 1950.f; 
    float mass = volume * rho0;
    float stiffness = 1000.f;	  
    float dt = 0.0006f;	 

    float DOMAIN_WIDTH = 20.f; // 1024
    float DOMAIN_HEIGHT = 10.f;  //1024
    float pad1;
    float pad2;

    float theta = 30.f * (float)M_PI / 180.f;       //* angle of repose
    float sigma = 0.25f;                            //* viscosity coefficient
    float alpha = 0.5f;                             //* viscosity constant
    float beta = 0.f;                               //* cohesion intensity
    
    float C = 0.f;                                  //* maximum cohesion
    float dragCoefficient = 0.47f;                  // Sphere Reynoldsnumber 10^6
    float rhoAir = 1293.f;                             //* Air density
    float pad3;
};


extern SPHSettings settings;

#endif