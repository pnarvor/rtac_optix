#include <optix_helpers/samples/materials.h>

namespace optix_helpers { namespace samples { namespace materials {

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


}; //namespace materials
}; //namespace samples
}; //namespace optix_helpers

