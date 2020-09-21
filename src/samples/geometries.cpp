#include <optix_helpers/samples/geometries.h>

#include <rtac_base/types/Mesh.h>

namespace optix_helpers { namespace samples { namespace geometries {

using Mesh = rtac::types::Mesh<float,uint32_t,3>;

GeometryTriangles cube(const Context& context, float scale)
{
    GeometryTriangles cube(context, false, true, true);
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
                          scale,  scale, -scale,
                          scale, -scale, -scale,
                         -scale, -scale, -scale,
                         -scale,  scale, -scale,
                          scale,  scale, -scale};
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
    
    cube->geometry()->setAttributeProgram(*context->create_program(Source(R"(
    #include <optix.h>
    #include <optixu/optixu_math_namespace.h>
    using namespace optix;

    rtBuffer<float3> vertex_buffer;
    rtBuffer<float3> normal_buffer;
    rtBuffer<float2> texcoord_buffer;
    
    rtDeclareVariable(float3, n, attribute normal,);
    rtDeclareVariable(float2, uv, attribute texture_coordinates,);
    
    RT_PROGRAM void cube_attributes()
    {
        const unsigned int primitiveIndex = rtGetPrimitiveIndex();
        const float2 barycentrics = rtGetTriangleBarycentrics();
        float3 w = make_float3(1.0 - barycentrics.x - barycentrics.y,
                               barycentrics.x,
                               barycentrics.y);

        n = w.x*normal_buffer[3*primitiveIndex]
          + w.y*normal_buffer[3*primitiveIndex + 1]
          + w.z*normal_buffer[3*primitiveIndex + 2];
        uv = w.x*texcoord_buffer[3*primitiveIndex]
           + w.y*texcoord_buffer[3*primitiveIndex + 1]
           + w.z*texcoord_buffer[3*primitiveIndex + 2];
    }
    )", "cube_attributes")));
    return cube;
}

GeometryTriangles square(const Context& context, float scale)
{
    GeometryTriangles square(context, false, true, true);
    float points[18] = {//x+ faces
                        -scale, -scale, 0.0,
                         scale, -scale, 0.0,
                         scale,  scale, 0.0,
                        -scale, -scale, 0.0,
                         scale,  scale, 0.0,
                        -scale,  scale, 0.0};
    float normals[18] = {0,0,1,
                         0,0,1,
                         0,0,1,
                         0,0,1,
                         0,0,1,
                         0,0,1};
    float texCoords[12] = {//x+ faces
                           0,0,
                           1,0,
                           1,1,
                           0,0,
                           1,1,
                           0,1};
    square->set_points(6, points);
    square->set_normals(6, normals);
    square->set_texture_coordinates(6, texCoords);
    square->geometry()->setPrimitiveCount(2);

    square->geometry()->setAttributeProgram(*context->create_program(Source(R"(
    #include <optix.h>
    #include <optixu/optixu_math_namespace.h>
    using namespace optix;

    rtBuffer<float3> vertex_buffer;
    rtBuffer<float3> normal_buffer;
    rtBuffer<float2> texcoord_buffer;
    
    rtDeclareVariable(float3, n, attribute normal,);
    rtDeclareVariable(float2, uv, attribute texture_coordinates,);
    
    RT_PROGRAM void square_attributes()
    {
        const unsigned int primitiveIndex = rtGetPrimitiveIndex();
        const float2 barycentrics = rtGetTriangleBarycentrics();
        float3 w = make_float3(1.0 - barycentrics.x - barycentrics.y,
                               barycentrics.x,
                               barycentrics.y);

        n = w.x*normal_buffer[3*primitiveIndex]
          + w.y*normal_buffer[3*primitiveIndex + 1]
          + w.z*normal_buffer[3*primitiveIndex + 2];
        uv = w.x*texcoord_buffer[3*primitiveIndex]
           + w.y*texcoord_buffer[3*primitiveIndex + 1]
           + w.z*texcoord_buffer[3*primitiveIndex + 2];
    }
    )", "square_attributes")));
    return square;
}

Geometry sphere(const Context& context, float radius)
{
    Program intersection(context->create_program(Source(R"(
    #include <optix.h>
    #include <optix_math.h>

    #include <optix_helpers/maths.h>

    rtDeclareVariable(optix::Ray, ray, rtCurrentRay,);
    
    rtDeclareVariable(float, radius,,);

    rtDeclareVariable(float3, n, attribute normal,);
    rtDeclareVariable(float2, uv, attribute texture_coordinates,);
    
    RT_PROGRAM void intersection(int)
    {
        //// Intersection of sphere and ray
        float tmin, tmax;
        if(!sphere_intersection(ray, radius, tmin, tmax))
            return;
        float3 p;
        p = ray.origin + tmin*ray.direction;
        if(rtPotentialIntersection(tmin)) {
            n = sphere_normal(p);
            uv.x = 0.5 * (atan2f(n.y,n.x) / M_PIf + 1.0f);
            uv.y = atan2f(n.z, sqrtf(n.x*n.x+n.y*n.y)) / M_PIf;
            rtReportIntersection(0);
        }
        p = ray.origin + tmax*ray.direction;
        if(rtPotentialIntersection(tmax)) {
            n = sphere_normal(p);
            uv.x = 0.5 * (atan2f(n.y,n.x) / M_PIf + 1.0f);
            uv.y = atan2f(n.z, sqrtf(n.x*n.x+n.y*n.y)) / M_PIf;
            rtReportIntersection(0);
        }
    }
    )", "intersection"), {maths::maths}));

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
    return Geometry(context, intersection, boundingbox, 1);
}


Geometry tube(const Context& context, float radius, float height)
{
    Program intersection(context->create_program(Source(R"(
    #include <optix.h>
    #include <optix_math.h>

    #include <optix_helpers/maths.h>

    rtDeclareVariable(optix::Ray, ray, rtCurrentRay,);
    
    rtDeclareVariable(float, radius,,);
    rtDeclareVariable(float, height,,);

    rtDeclareVariable(float3, n, attribute normal,);
    rtDeclareVariable(float2, uv, attribute texture_coordinates,);
    
    RT_PROGRAM void intersection(int)
    {
        float t1, t2;
        if(!tube_intersection(ray, radius, t1, t2))
            return;
        
        float3 p;
        p = ray.origin + t1 * ray.direction;
        if(abs(p.z) < height && rtPotentialIntersection(t1)) {
            n = tube_normal(p);
            uv.x = 0.5f * (atan2f(n.y,n.x) / M_PIf + 1.0f);
            uv.y = 0.5f * (p.z / height + 1.0f);
            rtReportIntersection(0);
        }
        p = ray.origin + t2 * ray.direction;
        if(abs(p.z) < height && rtPotentialIntersection(t2)) {
            n = tube_normal(p);
            uv.x = 0.5f * (atan2f(n.y,n.x) / M_PIf + 1.0f);
            uv.y = 0.5f * (p.z / height + 1.0f);
            rtReportIntersection(0);
        }
    }
    )", "intersection"), {maths::maths}));

    Program boundingbox(context->create_program(Source(R"(
    #include <optix.h>
    
    rtDeclareVariable(float, radius,,);
    rtDeclareVariable(float, height,,);
    
    RT_PROGRAM void bounds(int, float bbox[6])
    {
        bbox[0] = -radius;
        bbox[1] = -radius;
        bbox[2] = -height;
        bbox[3] =  radius;
        bbox[4] =  radius;
        bbox[5] =  height;
    }
    )", "bounds")));
    
    (*intersection)["radius"]->setFloat(radius);
    (*intersection)["height"]->setFloat(height);
    (*boundingbox)["radius"]->setFloat(radius);
    (*boundingbox)["height"]->setFloat(height);
    return Geometry(context, intersection, boundingbox, 1);
}

Geometry parabola(const Context& context, float a, float b, float height)
{
    Program intersection(context->create_program(Source(R"(
    #include <optix.h>
    #include <optix_math.h>

    #include <optix_helpers/maths.h>

    rtDeclareVariable(optix::Ray, ray, rtCurrentRay,);
    
    rtDeclareVariable(float, a,,);
    rtDeclareVariable(float, b,,);
    rtDeclareVariable(float, bottom,,);
    rtDeclareVariable(float, top,,);

    rtDeclareVariable(float3, n, attribute normal,);
    rtDeclareVariable(float2, uv, attribute texture_coordinates,);
    
    RT_PROGRAM void intersection(int)
    {
        float t1, t2;
        if(!parabola_intersection(ray, a, b, t1, t2))
            return;
        
        float3 p;
        p = ray.origin + t1 * ray.direction;
        if(p.z >= bottom && p.z <= top && rtPotentialIntersection(t1)) {
            n = parabola_normal(p,a,b);
            uv.x = 0.5f * (atan2f(p.y,p.x) / M_PIf + 1.0f);
            uv.y = (p.z - bottom) / (top - bottom);
            rtReportIntersection(0);
        }
        p = ray.origin + t2 * ray.direction;
        if(p.z >= bottom && p.z <= top && rtPotentialIntersection(t2)) {
            n = parabola_normal(p,a,b);
            uv.x = 0.5f * (atan2f(p.y,p.x) / M_PIf + 1.0f);
            uv.y = (p.z - bottom) / (top - bottom);
            rtReportIntersection(0);
        }
    }
    )", "intersection"), {maths::maths}));

    Program boundingbox(context->create_program(Source(R"(
    #include <optix.h>
    
    rtDeclareVariable(float, radius,,);
    rtDeclareVariable(float, bottom,,);
    rtDeclareVariable(float, top,,);
    
    RT_PROGRAM void bounds(int, float bbox[6])
    {
        bbox[0] = -radius;
        bbox[1] = -radius;
        bbox[2] =  bottom;
        bbox[3] =  radius;
        bbox[4] =  radius;
        bbox[5] =     top;
    }
    )", "bounds")));
    
    if(fabs(a) < 1.0e-6) {
        throw std::runtime_error("a parameter too low for hyperbola");
    }
    if(a >= 0.0f) {
        (*boundingbox)["bottom"]->setFloat(b);
        (*boundingbox)["top"]->setFloat(b + height);
        (*intersection)["bottom"]->setFloat(b);
        (*intersection)["top"]->setFloat(b + height);
    }
    else {
        (*boundingbox)["bottom"]->setFloat(b - height);
        (*boundingbox)["top"]->setFloat(b);
        (*intersection)["bottom"]->setFloat(b - height);
        (*intersection)["top"]->setFloat(b);
    }
    (*boundingbox)["radius"]->setFloat(sqrt(fabs(height / a)));
    
    (*intersection)["a"]->setFloat(a);
    (*intersection)["b"]->setFloat(b);
    return Geometry(context, intersection, boundingbox, 1);
}


//GeometryTriangles indexed_cube(const Context& context, float scale)
//{
//    return context->create_geometry_triangles(Mesh::cube(scale));
//}


}; //namespace geometries
}; //namespace samples
}; //namespace optix_helpers
