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
    optixSetPayload_0(25);
    optixSetPayload_1(25);
    optixSetPayload_2(25);
}

extern "C" __global__ void __closesthit__sbt_test()
{
    auto recordData = reinterpret_cast<ClosestHitData*>(optixGetSbtDataPointer());
    
    // There is 3 uv coordinates for each triangle (on per vertex) so
    // memory offset is 3*primitiveIndex;
    float2* vertexUVs = recordData->uvCoords + 3*optixGetPrimitiveIndex();

    float2 b = optixGetTriangleBarycentrics();
    float2 uv = (1.0f - b.x - b.y) * vertexUVs[0]
              +                b.x * vertexUVs[1]
              +                b.y * vertexUVs[2];

    float4 color = tex2D<float4>(recordData->texObject, uv.x, uv.y);

    optixSetPayload_0((uint32_t)255*color.x);
    optixSetPayload_1((uint32_t)255*color.y);
    optixSetPayload_2((uint32_t)255*color.z);
}


