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

Material lambert(const Context& context, const raytypes::RGB& rayType,
                 const std::array<float,3>& light, const std::array<float,3>& color)
{
    Source closestHit(R"(
    #include <optix.h>
    #include <optix_math.h>
    using namespace optix;
    
    #include <rays/RGB.h>
    
    rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
    rtDeclareVariable(float, tHit, rtIntersectionDistance,);
    rtDeclareVariable(raytypes::RGB, rayPayload, rtPayload, );
    
    rtDeclareVariable(float3, n_object, attribute normal,);

    rtDeclareVariable(float3, light,,);
    rtDeclareVariable(float3, color,,);
    
    RT_PROGRAM void closest_hit_perfect_mirror()
    {
        float3 n = rtTransformNormal(RT_OBJECT_TO_WORLD, n_object);
        float3 hitPoint = ray.origin + tHit*ray.direction;
        float3 v = normalize(light - hitPoint);

        //rayPayload.color = color*max(dot(n,v), 0.0f);
        rayPayload.color = color*abs(dot(n,v));
    }
    
    )", "closest_hit_perfect_mirror");
    Material material = context->create_material();
    Program program = context->create_program(closestHit, {rayType->definition()});
    
    (*program)["light"]->setFloat(make_float3(light));
    (*program)["color"]->setFloat(make_float3(color));

    material->add_closest_hit_program(rayType, program);
    return material;
}

Material barycentrics(const Context& context, const raytypes::RGB& rayType)
{
    Source closestHit(R"(
    #include <optix.h>
    using namespace optix;
    
    #include <rays/RGB.h>
    
    rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
    rtDeclareVariable(raytypes::RGB, rayPayload, rtPayload, );

    rtDeclareVariable(float2, weights, attribute rtTriangleBarycentrics,);
    
    RT_PROGRAM void closest_hit_white()
    {
        rayPayload.color.x = 1.0f - weights.x - weights.y;
        rayPayload.color.y = weights.x;
        rayPayload.color.z = weights.y;
    }
    
    )", "closest_hit_white");
    
    Material material(context->create_material());
    material->add_closest_hit_program(rayType,
        context->create_program(closestHit, {rayType->definition()}));
    return material;
}

TexturedMaterial checkerboard(const Context& context, const raytypes::RGB& rayType,
                              const std::array<uint8_t,3>& color1,
                              const std::array<uint8_t,3>& color2,
                              size_t width, size_t height)
{
    Source closestHit(R"(
    #include <optix.h>
    using namespace optix;
    
    #include <rays/RGB.h>
    
    rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
    rtDeclareVariable(raytypes::RGB, rayPayload, rtPayload, );

    rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> inTexture;
    rtDeclareVariable(float2, uv, attribute texture_coordinates,);
    
    RT_PROGRAM void closest_hit_texture()
    {
        float4 c = tex2D(inTexture, uv.x, uv.y);
        rayPayload.color = make_float3(c.x,c.y,c.z);
    }
    
    )", "closest_hit_texture");
    TexturedMaterial material(
        (*context)->createMaterial(),
        textures::checkerboard(context, "inTexture", color1, color2, width, height));
    material->add_closest_hit_program(rayType,
        context->create_program(closestHit, {rayType->definition()}));
    return material;
}

Material perfect_mirror(const Context& context, const raytypes::RGB& rayType)
{
    Source closestHit(R"(
    #include <optix.h>
    #include <optix_math.h>
    using namespace optix;

    #include <optix_helpers/maths.h>
    
    #include <rays/RGB.h>

    rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
    rtDeclareVariable(float, tHit, rtIntersectionDistance,);
    rtDeclareVariable(raytypes::RGB, rayPayload, rtPayload, );
    
    rtDeclareVariable(rtObject, topObject,,);

    rtDeclareVariable(float3, n_object, attribute normal,);
    
    RT_PROGRAM void closest_hit_perfect_mirror()
    {
        float3 n = rtTransformNormal(RT_OBJECT_TO_WORLD, n_object);
        float3 hitPoint = ray.origin + tHit*ray.direction;
        Ray reflectedRay(hitPoint,
                         optix::reflect(ray.direction, n),
                         //reflection(ray.direction, n),
                         ray.ray_type, 
                         1.0e-4);
                         //ray.tmin, ray.tmax);

        raytypes::RGB reflectedPayload;
        rtTrace(topObject, reflectedRay, reflectedPayload);
        //rayPayload = reflectedPayload;
        rayPayload.color = 0.9f*reflectedPayload.color;
    }
    
    )", "closest_hit_perfect_mirror");
    Material material = context->create_material();
    material->add_closest_hit_program(rayType,
        context->create_program(closestHit, {rayType->definition(), maths::maths}));
    return material;
}

    return material;
}

}; //namespace materials
}; //namespace samples
}; //namespace optix_helpers

