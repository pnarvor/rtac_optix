#include <optix_helpers/samples/materials.h>

namespace optix_helpers { namespace samples { namespace materials {

Material::Ptr rgb(const Context::ConstPtr& context, const raytypes::RGB& rayType,
                  const std::array<float,3>& color)
{
    auto closestHit = context->create_program(Source::New(R"(
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
    
    )", "closest_hit_rgb"), {rayType.definition()});
    (*closestHit)["rgbColor"]->setFloat(make_float3(color));
    
    auto material = Material::New(context);
    material->add_closest_hit_program(rayType, closestHit);
    return material;
}

Material::Ptr white(const Context::ConstPtr& context, const raytypes::RGB& rayType)
{
    auto closestHit = Source::New(R"(
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
    
    auto material = Material::New(context);
    material->add_closest_hit_program(rayType,
        context->create_program(closestHit, {rayType.definition()}));
    return material;
}

Material::Ptr black(const Context::ConstPtr& context, const raytypes::RGB& rayType)
{
    return rgb(context, rayType, {0.0,0.0,0.0});
}

Material::Ptr red(const Context::ConstPtr& context, const raytypes::RGB& rayType)
{
    return rgb(context, rayType, {1.0,0.0,0.0});
}

Material::Ptr green(const Context::ConstPtr& context, const raytypes::RGB& rayType)
{
    return rgb(context, rayType, {0.0,1.0,0.0});
}

Material::Ptr blue(const Context::ConstPtr& context, const raytypes::RGB& rayType)
{
    return rgb(context, rayType, {0.0,0.0,1.0});
}

Material::Ptr lambert(const Context::ConstPtr& context, const raytypes::RGB& rayType,
                 const std::array<float,3>& light, const std::array<float,3>& color)
{
    auto closestHit = Source::New(R"(
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
    auto material = Material::New(context);
    auto program = context->create_program(closestHit, {rayType.definition()});
    
    (*program)["light"]->setFloat(make_float3(light));
    (*program)["color"]->setFloat(make_float3(color));

    material->add_closest_hit_program(rayType, program);
    return material;
}

Material::Ptr barycentrics(const Context::ConstPtr& context, const raytypes::RGB& rayType)
{
    auto closestHit = Source::New(R"(
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
    
    auto material = Material::New(context);
    material->add_closest_hit_program(rayType,
        context->create_program(closestHit, {rayType.definition()}));
    return material;
}

TexturedMaterial::Ptr checkerboard(const Context::ConstPtr& context,
                                   const raytypes::RGB& rayType,
                                   const std::array<uint8_t,3>& color1,
                                   const std::array<uint8_t,3>& color2,
                                   size_t width, size_t height)
{
    auto closestHit = Source::New(R"(
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
    auto material = TexturedMaterial::New(context,
        textures::checkerboard(context, "inTexture", color1, color2, width, height));
    material->add_closest_hit_program(rayType,
        context->create_program(closestHit, {rayType.definition()}));
    return material;
}

Material::Ptr perfect_mirror(const Context::ConstPtr& context, const raytypes::RGB& rayType)
{
    auto closestHit = Source::New(R"(
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
    auto material = Material::New(context);
    material->add_closest_hit_program(rayType,
        context->create_program(closestHit, {rayType.definition(), maths::maths}));
    return material;
}

Material::Ptr perfect_refraction(const Context::ConstPtr& context, const raytypes::RGB& rayType,
                                 float refractiveIndex)
{
    auto closestHit = Source::New(R"(
    #include <optix.h>
    #include <optix_math.h>
    using namespace optix;
    
    #include <optix_helpers/maths.h>
    
    #include <rays/RGB.h>
    
    rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
    rtDeclareVariable(float, tHit, rtIntersectionDistance,);
    rtDeclareVariable(raytypes::RGB, rayPayload, rtPayload, );
    rtDeclareVariable(float3, n_object, attribute normal,);
    
    rtDeclareVariable(rtObject, topObject,,);

    rtDeclareVariable(float, refractiveIndex,,);

    RT_PROGRAM void closest_hit_perfect_refraction()
    {
        float3 n = rtTransformNormal(RT_OBJECT_TO_WORLD, n_object);
        float3 hitPoint = ray.origin + tHit*ray.direction;
        float3 refractedDir;
        optix::refract(refractedDir, ray.direction, n, refractiveIndex);
        //refractedDir = refraction(ray.direction, n, refractiveIndex);
        Ray refractedRay(hitPoint,
                         refractedDir,
                         ray.ray_type, 
                         1.0e-4f);

        raytypes::RGB refractedPayload;
        rtTrace(topObject, refractedRay, refractedPayload);
        rayPayload.color = 0.9*refractedPayload.color;
    }
    
    )", "closest_hit_perfect_refraction");
    auto material = Material::New(context);
    auto program = context->create_program(closestHit, 
                                           {rayType.definition(), maths::maths});
    (*program)["refractiveIndex"]->setFloat(refractiveIndex);
    material->add_closest_hit_program(rayType, program);
    return material;
}

}; //namespace materials
}; //namespace samples
}; //namespace optix_helpers

