#pragma once 
#ifndef HEADER_H
#define HEADER_H
#define _USE_MATH_DEFINES
#include <math.h>
#include <glm/glm.hpp>

extern bool simulationRunning;

//? sphere Volume in dm^3 * density
//   float MASS = (4.f / 3.f * (float) M_PI * (particleRadius * particleRadius * particleRadius)) * rho0; 
//? sphere Volume in 2D 
// float mass = volume * rho0; //* kg

#define particleRadius 0.001f //* m   //? 0.063 bis 2 mm // 1dm = 10cm = 100mm
#define volume (float) M_PI * (particleRadius * particleRadius)

struct SPHSettings{
    glm::vec2 G = glm::vec2(0.f, -9.81f); //* m/s^2
    float rho0 = 1.5f; 
    float kernelRadius = 0.3f; 

    float mass = 1.f;
    float stiffness = 50.f;	  
    float dt = 0.000f;	  
    float DOMAIN_WIDTH = 60.f; 

    float DOMAIN_HEIGHT = 30.f; 
    float theta = 30.f * (float)M_PI / 180.f;     //* angle of repose
    float sigma = 0.25f;    //* viscosity coefficient
    float beta = 0.f;       //* cohesion intensity

    float C = 0.f;          //* maximum cohesion
    float alpha = 0.5f;     //* viscosity constant
    float pad1 = 0.f;
    float pad2 = 0.f;
};


extern SPHSettings settings;

#endif