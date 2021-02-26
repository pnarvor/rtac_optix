#include "materials_test.h"

extern "C" {
    __constant__ Params params;
};

extern "C" __global__ void __raygen__materials_test()
{
    uint3 idx  = optixGetLaunchIndex();
    uint3 dims = optixGetLaunchDimensions();

    float3 rayOrigin, rayDirection;
    params.cam.compute_ray(idx, dims, rayOrigin, rayDirection);
    uint32_t p0, p1, p2;
    optixTrace(params.topObject,
               rayOrigin, rayDirection,
               0.0f , 1.0e8f, 0.0f, 255,
               OPTIX_RAY_FLAG_NONE,
               0, 1, 0, p0, p1, p2);
    params.imgData[params.width*idx.y + idx.x] = make_uchar3(p0,p1,p2);
}

extern "C" __global__ void __miss__materials_test()
{
    optixSetPayload_0(0);
    optixSetPayload_1(0);
    optixSetPayload_2(0);
}

extern "C" __global__ void __closesthit__materials_test()
{
    auto data = reinterpret_cast<const HitData*>(optixGetSbtDataPointer());
    optixSetPayload_0(data->color.x);
    optixSetPayload_1(data->color.y);
    optixSetPayload_2(data->color.z);
}



