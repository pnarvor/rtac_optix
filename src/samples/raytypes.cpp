#include <optix_helpers/samples/raytypes.h>

#include <optix_helpers/utils.h>

namespace optix_helpers { namespace samples { namespace raytypes {

RayType::Index RGB::typeIndex = RayType::uninitialized;

const Source::ConstPtr RGB::typeDefinition = Source::New(R"(
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

RGB::RGB(const Context::ConstPtr& context) :
    RayType(context->instanciate_raytype<RGB>())
{}

Program::Ptr RGB::rgb_miss_program(const Context::ConstPtr& context,
                                   const std::array<float,3>& color)
{
    auto program = context->create_program(Source::New(R"(
    #include <optix.h>
    using namespace optix;
    
    #include <rays/RGB.h>
    
    rtDeclareVariable(raytypes::RGB, rayPayload, rtPayload, );
    rtDeclareVariable(float3, missColor,,);
    
    RT_PROGRAM void rgb_miss()
    {
        rayPayload.color.x = missColor.x;
        rayPayload.color.y = missColor.y;
        rayPayload.color.z = missColor.z;
    }
    )", "rgb_miss"),
    {RGB::typeDefinition});
    (*program)["missColor"]->setFloat(make_float3(color));

    return program;
}

Program::Ptr RGB::black_miss_program(const Context::ConstPtr& context)
{
    return context->create_program(Source::New(R"(
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
    {RGB::typeDefinition});
}

}; //namespace raytypes
}; //namespace samples
}; //namespace optix_helpers


