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
    RgbRay::to_registers(data->color);
}

extern "C" __global__ void __miss__autosbt_shadow()
{
    ShadowRay::to_registers({-1.0f});
}

extern "C" __global__ void __closesthit__autosbt_shadow()
{
    ShadowRay::to_registers({optixGetRayTmax()});
}
