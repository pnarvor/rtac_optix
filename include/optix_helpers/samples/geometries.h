#ifndef _DEF_OPTIX_HELPERS_SAMPLES_GEOMETRIES_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_GEOMETRIES_H_

#include <rtac_base/types/Mesh.h>

#include <optix_helpers/Context.h>
#include <optix_helpers/Program.h>
#include <optix_helpers/Geometry.h>
#include <optix_helpers/GeometryTriangles.h>

#include <optix_helpers/samples/maths.h>

namespace optix_helpers { namespace samples { namespace geometries {

GeometryTriangles::Ptr cube(const Context::ConstPtr& context, float scale = 1.0);
Geometry::Ptr          sphere(const Context::ConstPtr& context, float radius = 1.0);
GeometryTriangles::Ptr square(const Context::ConstPtr& context, float scale = 1.0);
Geometry::Ptr          tube(const Context::ConstPtr& context, float radius = 1.0, 
                       float height = 1.0);

template <typename Tp, typename Tf>
GeometryTriangles::Ptr mesh(const Context::ConstPtr& mesh,
                            const rtac::types::Mesh<Tp,Tf,3>& m);

Geometry::Ptr parabola(const Context::ConstPtr& context, float a = 1.0, float b = 1.0,
                  float height = 1.0);

// Implementation
template <typename Tp, typename Tf>
GeometryTriangles::Ptr mesh(const Context::ConstPtr& context,
                            const rtac::types::Mesh<Tp,Tf,3>& m)
{
    auto res = GeometryTriangles::New(context, m);

    res->geometry()->setAttributeProgram(*context->create_program(Source::New(R"(
    #include <optix.h>
    #include <optixu/optixu_math_namespace.h>
    using namespace optix;

    rtBuffer<float3> vertex_buffer;
    rtBuffer<uint3>  index_buffer;
    
    rtDeclareVariable(float3, n, attribute normal,);
    rtDeclareVariable(float2, uv, attribute texture_coordinates,);
    
    RT_PROGRAM void cube_attributes()
    {
        const unsigned int primitiveIndex = rtGetPrimitiveIndex();
        //const float2 barycentrics = rtGetTriangleBarycentrics();
        
        float3 p0 = vertex_buffer[index_buffer[primitiveIndex].x];
        float3 p1 = vertex_buffer[index_buffer[primitiveIndex].y];
        float3 p2 = vertex_buffer[index_buffer[primitiveIndex].z];

        n = normalize(cross(p1-p0, p2-p0));

        uv = make_float2(0.0f,0.0f);
    }
    )", "cube_attributes")));
    return res;
}

}; //namespace geometries
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_GEOMETRIES_H_

