
#include <glm/glm.hpp>
#include "utils.h"

#include <tmd/TriangleMeshDistance.h>

struct AABB{
    glm::vec3 min = glm::vec3(0.0);
    glm::vec3 max = glm::vec3(0.0);

    glm::vec3 size(){
        return max - min;
    };
    glm::vec3 center(){
        return (max + min) / 2.f;
    };
};

struct RigidBody2D{
    bool active = false; // states if object is influenced by forces
    bool invert = false; // states uf the sdf should be inverted
    glm::vec3 position = glm::vec3(0.0);
    glm::vec3 scale = glm::vec3(1.0);
    AABB aabb;
    virtual glm::vec3 signedDistanceGradient(glm::vec3 position) = 0; // calculates the signed distance and direction 
    virtual float signedDistance(glm::vec3 position) = 0; // calculates the signed distance 
};

struct Box3D : public RigidBody2D{
    glm::vec3 halfSize;
    inline Box3D(glm::vec3 halfSize) : halfSize(halfSize) {
        aabb.min = -halfSize;
        aabb.max = halfSize;
    };
    glm::vec3 signedDistanceGradient(glm::vec3 p) override {
        glm::vec3 q = glm::abs(p) - halfSize;
        glm::vec3 n = (q.x > q.y ? glm::vec3(std::sign(p.x), 0, 0) : q.y > q.z ?  glm::vec3(0, std::sign(p.y), 0) : glm::vec3(0, 0, std::sign(p.z)));
        return n * glm::length(glm::max(q,glm::vec3(0.0))) + std::min(std::max(q.x, std::max(q.y, q.z)), 0.f);
    };
   
    float signedDistance(glm::vec3 p) override {
        glm::vec3 q = glm::abs(p) - halfSize;
        return glm::length(glm::max(q,glm::vec3(0.0))) + std::min(std::max(q.x, std::max(q.y, q.z)), 0.f);
    };
};

struct Plane3D : public RigidBody2D{
    glm::vec3 normal;
    float h; 
    inline Plane3D(glm::vec3 normal, float h) : normal(normal), h(h) {
        aabb.min = glm::normalize(normal) * h;
        aabb.max = glm::normalize(normal) * h;
        // aabb.min = glm::vec3(0);
        // aabb.max = glm::vec3(0);
    };
    glm::vec3 signedDistanceGradient(glm::vec3 p) override {
        return normal * (glm::dot(p, normal) + h);
    };
    float signedDistance(glm::vec3 p) override {
        return (glm::dot(p, normal) + h);
    };
};


struct Mesh3D : public RigidBody2D{

    std::vector<std::array<double, 3>> vertices;
    std::vector<std::array<int, 3>> triangles;
    tmd::TriangleMeshDistance mesh_distance;

    Mesh3D(std::string src);
    glm::vec3 signedDistanceGradient(glm::vec3 p) override {
        tmd::Result result = mesh_distance.signed_distance({ p.x, p.y, p.z });
        return p - glm::vec3(result.nearest_point[0], result.nearest_point[1], result.nearest_point[2]);
    };
    float signedDistance(glm::vec3 p) override {
        tmd::Result result = mesh_distance.signed_distance({ p.x, p.y, p.z });
        return result.distance;
    };
};
