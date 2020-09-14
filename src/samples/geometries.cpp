#include <optix_helpers/samples/geometries.h>

#include <rtac_base/types/Mesh.h>

namespace optix_helpers { namespace samples { namespace geometries {

using Mesh = rtac::types::Mesh<float,uint32_t,3>;

GeometryTriangles cube(const Context& context, float scale)
{
    return context->create_geometry_triangles(Mesh::cube(scale));
}

Geometry sphere(const Context& context, float radius)
{
    Program intersection(context->create_program(Source(R"(
#include <optix.h>
#include <optix_math.h>

rtDeclareVariable(optix::Ray, ray, rtCurrentRay,);

rtDeclareVariable(float, radius,,);

RT_PROGRAM void intersection(int)
{
    // Intersection of sphere and ray
    // assuming a = 1.0
    float a = 1.0; // = dot(ray.direction, ray.direction);
    float b = dot(ray.origin, ray.direction);
    float c = dot(ray.origin, ray.origin) - radius*radius;
    float delta = 4*(b*b - c);

    if(delta < 0.0f) return;

    float tmin = 0.5*(-b - sqrt(delta));
    float tmax = 0.5*(-b + sqrt(delta));
    if(tmin > 0.0f) {
        if(rtPotentialIntersection(tmin)) {
            rtReportIntersection(0);
        }
    }
    else if(tmax > 0.0f) {
        if(rtPotentialIntersection(tmax)) {
            rtReportIntersection(0);
        }
    }
}

    )", "intersection")));

    Program boundingbox(context->create_program(Source(R"(
#include <optix.h>

rtDeclareVariable(float, radius,,);

RT_PROGRAM void bounds(int, float bbox[6])
{
    bbox[0] = -radius;
    bbox[1] = -radius;
    bbox[2] = -radius;
    bbox[3] =  radius;
    bbox[4] =  radius;
    bbox[5] =  radius;
}
    )", "bounds")));
    
    (*intersection)["radius"]->setFloat(radius);
    (*boundingbox)["radius"]->setFloat(radius);
    return context->create_geometry(intersection, boundingbox, 1);
}

GeometryTriangles cube_with_attributes(const Context& context, float scale)
{
    GeometryTriangles cube = context->create_geometry_triangles(false, true, true);
    float points[108] = {//x+ faces
                          scale, -scale, -scale,
                          scale,  scale, -scale,
                          scale,  scale,  scale,
                          scale, -scale, -scale,
                          scale,  scale,  scale,
                          scale, -scale,  scale,
                         //x- faces
                         -scale, -scale, -scale,
                         -scale,  scale,  scale,
                         -scale,  scale, -scale,
                         -scale, -scale, -scale,
                         -scale, -scale,  scale,
                         -scale,  scale,  scale,
                         //y+ faces
                          scale,  scale, -scale,
                         -scale,  scale, -scale,
                         -scale,  scale,  scale,
                          scale,  scale, -scale,
                         -scale,  scale,  scale,
                          scale,  scale,  scale,
                         //y- faces
                          scale, -scale, -scale,
                         -scale, -scale,  scale,
                         -scale, -scale, -scale,
                          scale, -scale, -scale,
                          scale, -scale,  scale,
                         -scale, -scale,  scale,
                         //z+ faces
                         -scale, -scale,  scale,
                          scale, -scale,  scale,
                          scale,  scale,  scale,
                         -scale, -scale,  scale,
                          scale,  scale,  scale,
                         -scale,  scale,  scale,
                         //z- faces
                         -scale, -scale, -scale,
                          scale, -scale, -scale,
                          scale,  scale, -scale,
                         -scale, -scale, -scale,
                          scale,  scale, -scale,
                         -scale,  scale, -scale};
    float normals[108] = { 1, 0, 0,
                           1, 0, 0,
                           1, 0, 0,
                           1, 0, 0,
                           1, 0, 0,
                           1, 0, 0,
                          -1, 0, 0,
                          -1, 0, 0,
                          -1, 0, 0,
                          -1, 0, 0,
                          -1, 0, 0,
                          -1, 0, 0,
                           0, 1, 0,
                           0, 1, 0,
                           0, 1, 0,
                           0, 1, 0,
                           0, 1, 0,
                           0, 1, 0,
                           0,-1, 0,
                           0,-1, 0,
                           0,-1, 0,
                           0,-1, 0,
                           0,-1, 0,
                           0,-1, 0,
                           0, 0, 1,
                           0, 0, 1,
                           0, 0, 1,
                           0, 0, 1,
                           0, 0, 1,
                           0, 0, 1,
                           0, 0,-1,
                           0, 0,-1,
                           0, 0,-1,
                           0, 0,-1,
                           0, 0,-1,
                           0, 0,-1};
    float texCoords[72] = {//x+ faces
                           0,0,
                           1,0,
                           1,1,
                           0,0,
                           1,1,
                           0,1,
                           //x- faces
                           0,0,
                           1,1,
                           1,0,
                           0,0,
                           0,1,
                           1,1,
                           //y+ faces
                           0,0,
                           1,0,
                           1,1,
                           0,0,
                           1,1,
                           0,1,
                           //y- faces
                           0,0,
                           1,1,
                           1,0,
                           0,0,
                           0,1,
                           1,1,
                           //z+ faces
                           0,0,
                           1,0,
                           1,1,
                           0,0,
                           1,1,
                           0,1,
                           //z- faces
                           0,0,
                           1,1,
                           1,0,
                           0,0,
                           0,1,
                           1,1};
    cube->set_points(36, points);
    cube->set_normals(36, normals);
    cube->set_texture_coordinates(36, texCoords);
    cube->geometry()->setPrimitiveCount(12);
    return cube;
}


}; //namespace geometries
}; //namespace samples
}; //namespace optix_helpers
