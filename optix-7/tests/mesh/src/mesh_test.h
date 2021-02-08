#ifndef _RTAC_OPTIX_TESTS_MESH_TEST_H_
#define _RTAC_OPTIX_TESTS_MESH_TEST_H_

#include <rtac_optix/samples/PinholeCamera.h>

struct Params
{
    unsigned int width;
    unsigned int height;
    uchar3*      imagedata;
    rtac::optix::samples::PinholeCamera cam;
};

struct RaygenData
{
    //nothing here for now
};

struct MissData
{
    //nothing here for now
};

#endif //_RTAC_OPTIX_TESTS_MESH_TEST_H_
