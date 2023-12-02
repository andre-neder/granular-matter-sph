
#include <glm/glm.hpp>

namespace std{
    template <typename T> int sign(T val) {
        return (T(0) < val) - (val < T(0));
    }
}

struct AABB{
    glm::vec2 min;
    glm::vec2 max;
};

struct RigidBody2D{
    bool active = false; // states if object is influenced by forces
    bool invert = false; // states uf the sdf should be inverted
    glm::vec2 position = glm::vec2(0.0);
    glm::vec2 scale = glm::vec2(1.0);
    AABB aabb;
    virtual glm::vec2 signedDistanceGradient(glm::vec2 position) = 0; // calculates the signed distance and direction 
    virtual float signedDistance(glm::vec2 position) = 0; // calculates the signed distance 
};

struct Box2D : public RigidBody2D{
    glm::vec2 halfSize;
    inline Box2D(glm::vec2 halfSize) : halfSize(halfSize) {
        aabb.min = -halfSize;
        aabb.max = halfSize;
    };
    glm::vec2 signedDistanceGradient(glm::vec2 p) override {
        glm::vec2 q = glm::abs(p) - halfSize;
        glm::vec2 n = (q.x > q.y ?  glm::vec2(std::sign(p.x), 0) : glm::vec2(0, std::sign(p.y)));
        return n * glm::length(glm::max(q,glm::vec2(0.0))) + std::min(std::max(q.x, q.y), 0.f);
    };
    float signedDistance(glm::vec2 p) override {
        glm::vec2 q = glm::abs(p) - halfSize;
        return glm::length(glm::max(q,glm::vec2(0.0))) + std::min(std::max(q.x, q.y), 0.f);
    };
};

struct Line2D : public RigidBody2D{
    glm::vec2 normal;
    float h; 
    inline Line2D(glm::vec2 normal, float h) : normal(normal), h(h) {
        aabb.min = glm::vec2(0);
        aabb.max = glm::vec2(0);
    };
    glm::vec2 signedDistanceGradient(glm::vec2 p) override {
        return normal * (glm::dot(p, normal) + h);
    };
    float signedDistance(glm::vec2 p) override {
        return (glm::dot(p, normal) + h);
    };
};

