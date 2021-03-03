#include "autosbt_test.h"

extern "C" {
    __constant__ Params params;
}

extern "C" __global__ void __raygen__autosbt_test()
{
    auto idx  = optixGetLaunchIndex();
    auto dims = optixGetLaunchDimensions();

    float3 rayOrigin, rayDirection;
    params.cam.compute_ray(idx, dims, rayOrigin, rayDirection);

    RgbRay ray;
    ray.trace(params.topObject, rayOrigin, rayDirection);
    params.output[params.width*idx.y + idx.x] = ray;
}

extern "C" __global__ void __miss__autosbt_rgb()
{
    auto data = reinterpret_cast<const RgbMissData*>(optixGetSbtDataPointer());
    RgbRay::to_registers(data->color);
}

extern "C" __global__ void __closesthit__autosbt_rgb()
{
    auto data = reinterpret_cast<const RgbHitData*>(optixGetSbtDataPointer());
    
    // Calculating hit position
    float3 vertices[3];
    optixGetTriangleVertexData(optixGetGASTraversableHandle(), // current object
                               optixGetPrimitiveIndex(),       // current triangle
                               optixGetSbtGASIndex(),          // ?
                               optixGetRayTmax(),              // Ray time at hit
                               vertices);
    float2 tBary = optixGetTriangleBarycentrics(); // Hit position in the triangle
                                                   // in barycentrics coordinates
    float3 hitPosition = optixTransformPointFromObjectToWorldSpace(
        (1.0f - tBary.x - tBary.y) * vertices[0] +
                           tBary.x * vertices[1] +
                           tBary.y * vertices[2]);

    // sending a shadow ray towards light starting hitPosition
    float3 rayDirection = normalize(data->light - hitPosition);
    ShadowRay sray;
    sray.trace(params.topObject, hitPosition, rayDirection, 1.0e-4);
    if(sray.tHit < 0.0f) {
        // the shadow ray did not encounter any object. No shadow
        RgbRay::to_registers(data->color);
    }
    else {
        // the shadow ray did encounter an object. Shadow
        uchar3 c = data->color;
        c.x /= 8;
        c.y /= 8;
        c.z /= 8;
        RgbRay::to_registers(c);
    }
}

extern "C" __global__ void __miss__autosbt_shadow()
{
    ShadowRay::to_registers({-1.0f});
}

extern "C" __global__ void __closesthit__autosbt_shadow()
{
    ShadowRay::to_registers({optixGetRayTmax()});
}
