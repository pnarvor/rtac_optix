#ifndef _DEF_RTAC_OPTIX_TESTS_MATERIALS_TESTS_H_
#define _DEF_RTAC_OPTIX_TESTS_MATERIALS_TESTS_H_

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/Raytype.h>
#include <rtac_optix/samples/PinholeCamera.h>

struct Params {
    unsigned int width;
    unsigned int height;
    uchar3*      imgData;
    rtac::optix::samples::PinholeCamera cam;
    OptixTraversableHandle topObject;
};

struct RaygenData {};

using RGBPayload = uchar3;
struct MissData {
    RGBPayload color;
};
struct HitData {
    RGBPayload color;
};

using RGBRay = rtac::optix::Raytype<RGBPayload,0,1,0>;

#endif //_DEF_RTAC_OPTIX_TESTS_MATERIALS_TESTS_H_
