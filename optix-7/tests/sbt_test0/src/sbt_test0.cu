#include "sbt_test0.h"

extern "C" {
    __constant__ Params params;
}

extern "C" __global__ void __raygen__sbt_test()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    float3 rayOrigin, rayDirection;
    params.cam.compute_ray(idx, dim, rayOrigin, rayDirection);

    uint32_t p0, p1, p2;
    optixTrace(params.sceneTreeHandle,
               rayOrigin, rayDirection,
               0.0f, 1e8f, // tmin,tmax
               0.0f, // raytime
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,
               0,1,0, // SBT offset, stride, missSBToffset
               p0, p1, p2);
    params.imageData[params.width*idx.y + idx.x] = make_uchar3(p0,p1,p2);
}

extern "C" __global__ void __miss__sbt_test()
{
    optixSetPayload_0(255);
    optixSetPayload_1(0);
    optixSetPayload_2(0);
}

extern "C" __global__ void __closesthit__sbt_test()
{
    auto recordData = reinterpret_cast<ClosestHitData*>(optixGetSbtDataPointer());
    uchar4 color = tex2D<uchar4>(recordData->texObject, 0.6f,0.3f);

    optixSetPayload_0(color.x);
    optixSetPayload_1(color.y);
    optixSetPayload_2(color.z);

    //optixSetPayload_0(0);
    //optixSetPayload_1(0);
    //optixSetPayload_2(255);
}


