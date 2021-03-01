#ifndef _DEF_RTAC_OPTIX_TEST_AUTOSBT_TEST_H_
#define _DEF_RTAC_OPTIX_TEST_AUTOSBT_TEST_H_

#include <rtac_optix/RaytypeFactory.h>
using namespace rtac::optix;
#include <rtac_optix/samples/PinholeCamera.h>
using PinholeCamera = rtac::optix::samples::PinholeCamera;

struct Params {
    unsigned int width;
    unsigned int height;
    uchar3* output;
    PinholeCamera cam;
    OptixTraversableHandle topObject;
};

using RgbPayload = uchar3;
struct RgbMissData {
    RgbPayload color;
};
struct RgbHitData {
    RgbPayload color;
};

struct ShadowPayload {
    float tHit;
};
struct ShadowMissData {};
struct ShadowHitData  {};

using Raytypes  = RaytypeFactory<RgbPayload, ShadowPayload>;
using RgbRay    = Raytypes::Raytype<0>;
using ShadowRay = Raytypes::Raytype<1>;

#endif //_DEF_RTAC_OPTIX_TEST_AUTOSBT_TEST_H_
