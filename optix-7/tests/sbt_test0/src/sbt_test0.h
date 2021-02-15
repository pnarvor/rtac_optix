#ifndef _DEF_RTAC_OPTIX_TESTS_SBT_TEST_0_H_
#define _DEF_RTAC_OPTIX_TESTS_SBT_TEST_0_H_

#include <optix.h>

#include <rtac_optix/samples/PinholeCamera.h>

struct Params
{
    unsigned int width;
    unsigned int height;
    uchar3*      imageData;
    rtac::optix::samples::PinholeCamera cam;
    OptixTraversableHandle sceneTreeHandle;
};

struct RaygenData
{
    // nothing here for now
};

struct MissData
{
    // nothing here for now
};

struct ClosestHitData
{
    cudaTextureObject_t texObject;
    float2*             uvCoords;
};

#endif //_DEF_RTAC_OPTIX_TESTS_SBT_TEST_0_H_
