#pragma once 
#ifndef HEADER_H
#define HEADER_H
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <iostream>
#include <utility>
#include <random>
#include <queue>
#include <glm/glm.hpp>

extern bool simulationRunning;
extern bool simulationStepForward;
extern bool resetSimulation;
extern int pauseOnFrame;
extern int currentFrameCount;
extern float simulationSpeedFactor;

extern float subTimeStep;
extern int substeps;

struct SPHSettings{
    glm::vec4 g = glm::vec4(0.f, -9.81f, 0.f, 0.f);          //* m/s^2
    
    float r_LR = 0.25f;                                  //* m
    float h_LR = r_LR * 4;                              //* m
    float rho0 = 1450.f;                                //* kg/m^3
    float mass = (4.f / 3.f * (float) M_PI * (r_LR * r_LR * r_LR)) * rho0;                //* kg

    float maxCompression = 0.01f;	  
    float dt = 0.0006f;	                                //* s
    float DOMAIN_WIDTH = 12.f;                         //* m
    float DOMAIN_HEIGHT = 8.f;                        //* m

    float sleepingSpeed = 0.0005f;                      //* m/s
    float h_HR = r_LR * 3;
    float theta = 45.f * (float)M_PI / 180.f;           //* rad (angle of repose)
    float rhoAir = 1293.f;                              //* Air density   

    glm::vec4 windDirection = glm::vec4(0.0);           //* Wind Direction
                                        
    float dragCoefficient = 0.47f;                      //* Sphere Reynoldsnumber 10^6
    uint32_t n_HR = 7 * 10;                             //* number of HR lrParticles per LR particle 2D: (/5) 3D: (/7)      
    float scale_W = 15.f / (float)(M_PI * pow(h_LR, 6.0));
    float scale_GradW = 60.f / (float) (pow(h_LR, 6.0) * M_PI);    
    float A_LR = r_LR * r_LR * (float)M_PI;                               
    float v_max = ((2.f * mass * glm::length(g)) / (rhoAir * A_LR * dragCoefficient));       
    float maxTimestep=0.016f;                        
    float pad1;                        
};

extern SPHSettings settings;

struct UniformBufferObject {
    glm::mat4 model = glm::mat4(1.0f);
    glm::mat4 view = glm::mat4(1.0f);
    glm::mat4 proj = glm::mat4(1.0f);
};

template<typename T>
struct ShiftingArray{
    inline ShiftingArray(){};
    inline ShiftingArray(uint32_t size, T defaultValue){
        data = std::vector<T>(size, defaultValue);
    }
    void append(T value){
        std::shift_left(data.begin(), data.end(), 1);
        data.push_back(value);
    }
    T get(uint32_t i){
        return data.at(i);
    }
    private:
        std::vector<T> data;
   
};

struct SimulationMetrics{
    static const uint32_t MAX_VALUES_PER_METRIC = 100;
    ShiftingArray<float> averageDensityError = ShiftingArray(100, 0.f);
    ShiftingArray<int> iterationCount = ShiftingArray(100, 2);
};

extern SimulationMetrics simulationMetrics;


#endif