#ifndef _DEF_RTAC_OPTIX_TESTS_SBT_INDEXES_TESTS_0_H_
#define _DEF_RTAC_OPTIX_TESTS_SBT_INDEXES_TESTS_0_H_

#include <optix.h>

#include <rtac_optix/samples/PinholeCamera.h>

struct Params {
    unsigned int   width;
    unsigned int   height;
    unsigned char* output;
    rtac::optix::samples::PinholeCamera cam;
    OptixTraversableHandle topObject;
};

struct RaygenData {};
struct MissData {
    unsigned int value;
};

struct HitData {
    unsigned int value;
};

#endif //_DEF_RTAC_OPTIX_TESTS_SBT_INDEXES_TESTS_0_H_
