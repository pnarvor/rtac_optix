#include <iostream>

namespace cusample {

std::string alphaHeader(R"(

#ifndef _DEF_HEADER_ALPHA_H_
#define _DEF_HEADER_ALPHA_H_

#define ALPHA 0.f

#endif //_DEF_HEADER_ALPHA_H_

)");

std::string drawColor(R"(

#include <optix.h>
#include <optixu/optixu_math_namespace.h>

using namespace optix;

#include <alpha.h>
//#define ALPHA 0.f

rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtBuffer<float4, 2>   result_buffer;

rtDeclareVariable(float3,                draw_color, , );

RT_PROGRAM void draw_solid_color()
{
  result_buffer[launch_index] = make_float4(draw_color, ALPHA);
})");

std::string coloredRay(R"(

#ifndef _DEF_RAYPAYLOAD_COLORED_RAY_H_
#define _DEF_RAYPAYLOAD_COLORED_RAY_H_

struct ColoredRay
{
    float3 color;
};

#endif //_DEF_RAYTYPE_COLORED_RAY_H_

)");

std::string whiteMaterial(R"(

#include <optix.h>
using namespace optix;

#include <colored_ray.h>

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(ColoredRay, rayPayload, rtPayload, );

RT_PROGRAM void closest_hit_white()
{
    rayPayload.color.x = 1.0f;
    rayPayload.color.y = 1.0f;
    rayPayload.color.z = 1.0f;
}

)");

std::string sphere = R"(
#include <optix.h>
#include <optix_math.h>

rtDeclareVariable(optix::Ray, ray, rtCurrentRay,);

RT_PROGRAM void intersection(int)
{
    // Intersection of sphere and ray
    // assuming a = 1.0
    float a = 1.0; // = dot(ray.direction, ray.direction);
    float b = dot(ray.origin, ray.direction);
    float c = dot(ray.origin, ray.origin) - 1.0f;
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

RT_PROGRAM void bounds(int, float bbox[6])
{
    bbox[0] = -1;
    bbox[1] = -1;
    bbox[2] = -1;
    bbox[3] =  1;
    bbox[4] =  1;
    bbox[5] =  1;
}

)";

std::string rayGenOrtho = R"(

#include <optix.h>
using namespace optix;

#include <colored_ray.h>

rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );
rtDeclareVariable(rtObject, topObject,,);
rtBuffer<float, 2> render_buffer;

RT_PROGRAM void ortho_z()
{
    size_t2 s = render_buffer.size();
    float3 origin = make_float3((2.0f*launch_index.x) / s.x - 1.0f,
                                (2.0f*launch_index.y) / s.y - 1.0f,
                                -1.0f);
    float3 direction = make_float3(0.0f,0.0f,1.0f);

    ColoredRay payload;
    payload.color = make_float3(0.0f,0.0f,0.0f);
    optix::Ray ray(origin, direction, 0, .1f);

    rtTrace(topObject, ray, payload);
    render_buffer[launch_index] = payload.color.x;
    //render_buffer[launch_index] = ray.origin.y;
}

)";

std::string coloredMiss = R"(

#include <optix.h>
using namespace optix;

#include <colored_ray.h>

rtDeclareVariable(ColoredRay, rayPayload, rtPayload, );

RT_PROGRAM void black_miss()
{
    //rayPayload.color.x = 0.0f;
    //rayPayload.color.x = 1.0f;
    rayPayload.color.x = 0.5f;
    rayPayload.color.y = 0.0f;
    rayPayload.color.z = 0.0f;
}

)";
}; //namespace cusample


