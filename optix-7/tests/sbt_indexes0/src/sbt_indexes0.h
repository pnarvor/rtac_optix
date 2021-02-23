#ifndef _DEF_RTAC_OPTIX_TESTS_SBT_INDEXES_TESTS_0_H_
#define _DEF_RTAC_OPTIX_TESTS_SBT_INDEXES_TESTS_0_H_

#include <optix.h>

#include <rtac_optix/RayPayload.h>
#include <rtac_optix/RayFactory.h>

#include <rtac_optix/samples/PinholeCamera.h>

struct Params {
    unsigned int   width;
    unsigned int   height;
    //unsigned char* output;
    uchar3*        output;
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

// RayTypes tests
template <typename T>
struct RGBPayload {
    uchar3 color;
};
template <typename T>
struct RGBRay : public rtac::optix::RayPayload<RGBPayload<T>> {
    using PayloadType = rtac::optix::RayPayload<RGBPayload<T>>;

    //RGBRay() : PayloadType() {}
    //RGBRay(const RGBPayload<T>& payload) : PayloadType(payload) {}
};

struct ShadowPayload {
    bool hit;
};
struct ShadowRay : public rtac::optix::RayPayload<ShadowPayload> {
    using PayloadType = rtac::optix::RayPayload<ShadowPayload>;
};

using RayBuilder = rtac::optix::RayFactoryBase<RGBRay<uchar3>, ShadowRay>;



#endif //_DEF_RTAC_OPTIX_TESTS_SBT_INDEXES_TESTS_0_H_
