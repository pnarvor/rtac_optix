#ifndef _DEF_RTAC_OPTIX_TESTS_MATERIALS_TESTS_H_
#define _DEF_RTAC_OPTIX_TESTS_MATERIALS_TESTS_H_

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/samples/PinholeCamera.h>

struct Params {
    unsigned int width;
    unsigned int height;
    uchar3*      imgData;
    rtac::optix::samples::PinholeCamera cam;
    OptixTraversableHandle topObject;
};

struct RaygenData {};
struct MissData   {};

struct HitData {
    uchar3 color;
};

template <unsigned int IndexV>
struct RayType
{
    static constexpr unsigned int Index = IndexV;
};

#endif //_DEF_RTAC_OPTIX_TESTS_MATERIALS_TESTS_H_
