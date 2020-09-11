#include <optix_helpers/samples/materials.h>

namespace optix_helpers { namespace samples { namespace materials {

Material rgb(const Context& context, const raytypes::RGB& rayType,
             const std::array<float,3>& color)
{
    Program closestHit = context->create_program(Source(R"(
    #include <optix.h>
    using namespace optix;
    
    #include <rays/RGB.h>
    
    rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
    rtDeclareVariable(raytypes::RGB, rayPayload, rtPayload, );
    rtDeclareVariable(float3, rgbColor,,);
    
    RT_PROGRAM void closest_hit_rgb()
    {
        rayPayload.color = rgbColor;
    }
    
    )", "closest_hit_rgb"), {rayType->definition()});
    (*closestHit)["rgbColor"]->setFloat(make_float3(color));
    
    Material material(context->create_material());
    material->add_closest_hit_program(rayType, closestHit);
    return material;
}

Material white(const Context& context, const raytypes::RGB& rayType)
{
    Source closestHit(R"(
    #include <optix.h>
    using namespace optix;
    
    #include <rays/RGB.h>
    
    rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
    rtDeclareVariable(raytypes::RGB, rayPayload, rtPayload, );
    
    RT_PROGRAM void closest_hit_white()
    {
        rayPayload.color.x = 1.0f;
        rayPayload.color.y = 1.0f;
        rayPayload.color.z = 1.0f;
    }
    
    )", "closest_hit_white");
    
    Material white(context->create_material());
    white->add_closest_hit_program(rayType,
        context->create_program(closestHit, {rayType->definition()}));
    return white;
}

Material black(const Context& context, const raytypes::RGB& rayType)
{
    return rgb(context, rayType, {0.0,0.0,0.0});
}

Material red(const Context& context, const raytypes::RGB& rayType)
{
    return rgb(context, rayType, {1.0,0.0,0.0});
}

Material green(const Context& context, const raytypes::RGB& rayType)
{
    return rgb(context, rayType, {0.0,1.0,0.0});
}

Material blue(const Context& context, const raytypes::RGB& rayType)
{
    return rgb(context, rayType, {0.0,0.0,1.0});
}



}; //namespace materials
}; //namespace samples
}; //namespace optix_helpers

