#include <optix_helpers/samples/raytypes.h>

namespace optix_helpers { namespace samples { namespace raytypes {

const Source RGB::ray_definition(R"(
#ifndef _DEF_RAYPAYLOAD_RGB_H_
#define _DEF_RAYPAYLOAD_RGB_H_

namespace raytypes {

struct RGB
{
    float3 color;
};

};

#endif //_DEF_RAYPAYLOAD_RGB_H_
)", "rays/RGB.h");

RGB::RGB(const Context& context) :
    RayType(context->create_raytype(RGB::ray_definition))
{}

Program RGB::black_miss_program(const Context& context)
{
    return context->create_program(Source(R"(
    #include <optix.h>
    using namespace optix;
    
    #include <rays/RGB.h>
    
    rtDeclareVariable(raytypes::RGB, rayPayload, rtPayload, );
    
    RT_PROGRAM void black_miss()
    {
        rayPayload.color.x = 0.0f;
        rayPayload.color.y = 0.0f;
        rayPayload.color.z = 0.0f;
    }
    )", "black_miss"),
    {RGB::ray_definition});
}

}; //namespace raytypes
}; //namespace samples
}; //namespace optix_helpers


