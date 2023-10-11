#pragma once 
#ifndef HEADER_H
#define HEADER_H
#include <glm/glm.hpp>

extern bool simulationRunning;

//? sphere Volume in dm^3 * density
//   float MASS = (4.f / 3.f * (float) M_PI * (particleRadius * particleRadius * particleRadius)) * rhoRest; 
//? sphere Volume in 2D 
// float mass = volume * rhoRest; //* kg

#define particleRadius 0.001f //* m   //? 0.063 bis 2 mm // 1dm = 10cm = 100mm
#define volume (float) M_PI * (particleRadius * particleRadius)

struct SPHSettings{
    glm::vec2 G = glm::vec2(0.f, -9.81f); //* m/s^2
    float rhoRest = 1.5f; //* kg / m^3 -> 2D kg / m^2  //? 1.5 - 2.2 kg / dm^3 (1950kg/m^3)
    float kernelRadius = 0.3f; // *m	

    float mass = 1.f;
    float stiffness = 25.f;	  
    float dt = 0.000f;	  
    float DOMAIN_WIDTH = 9.f; //* m

    float DOMAIN_HEIGHT = 9.f; //* m
    float pad0, pad1, pad2;
};


extern SPHSettings settings;

#endif