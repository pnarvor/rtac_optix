/* 
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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


}; //namespace cusample


