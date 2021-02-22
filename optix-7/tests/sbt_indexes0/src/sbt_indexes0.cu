#include "sbt_indexes0.h"

extern "C" {
    __constant__ Params params;
};

extern "C" __global__ void __raygen__sbt_indexes0()
{
    auto idx = optixGetLaunchIndex();
    auto dim = optixGetLaunchDimensions();

    float3 rayOrigin, rayDirection;
    params.cam.compute_ray(idx, dim, rayOrigin, rayDirection);
    
    unsigned int p0;
    optixTrace(params.topObject,
               rayOrigin, rayDirection,
               0.0f, 1.0e8f,
               0.0f, 255, OPTIX_RAY_FLAG_NONE,
               16, // sbtOffset
               10, // sbtstride
               0, // misssbtstride
               p0);
    params.output[params.width*idx.y + idx.x] = p0;
}

extern "C" __global__ void __miss__sbt_indexes0()
{
    auto data = reinterpret_cast<const MissData*>(optixGetSbtDataPointer());
    optixSetPayload_0(data->value);
}

extern "C" __global__ void __closesthit__sbt_indexes0()
{
    auto data = reinterpret_cast<const HitData*>(optixGetSbtDataPointer());
    optixSetPayload_0(data->value);
}
