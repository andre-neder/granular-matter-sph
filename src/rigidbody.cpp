
#include "rigidbody.h"
#include "model.h"

Mesh3D::Mesh3D()
{
}

Mesh3D::Mesh3D(std::string src)
{  
        Model model = Model();
        model.load_from_glb(src.c_str());
        aabb.min = glm::vec3(FLT_MAX); 
        aabb.max = glm::vec3(-FLT_MAX);
        for(auto v : model._vertices){
            vertices.push_back(std::array<double, 3>{v.pos.x, v.pos.y, v.pos.z});
            aabb.min = glm::vec3(std::min(v.pos.x, aabb.min.x), std::min(v.pos.y, aabb.min.y), std::min(v.pos.z, aabb.min.z));
            aabb.max = glm::vec3(std::max(v.pos.x, aabb.max.x), std::max(v.pos.y, aabb.max.y), std::max(v.pos.z, aabb.max.z));
        }
        for (uint32_t i = 0; i < model._indices.size(); i+=3)
        {
            triangles.push_back(std::array<int, 3>{static_cast<int>(model._indices[i]), static_cast<int>(model._indices[i + 1]), static_cast<int>(model._indices[i + 2])});
        }
        mesh_distance = tmd::TriangleMeshDistance(vertices, triangles);
};
