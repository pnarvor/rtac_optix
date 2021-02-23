#ifndef _DEF_RTAC_OPTIX_RAY_FACTORY_H_
#define _DEF_RTAC_OPTIX_RAY_FACTORY_H_

#include <tuple>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_base/type_utils.h>
#include <rtac_base/cuda/utils.h>

namespace rtac { namespace optix {

// RayFactoryBase is intended to simplify the creation of ray types and their use
// in both optix device code and host code. The goal is to automatically manage
// the ray type indexes and strides.
// This simple objective in appearance actually implies that all the ray type
// indexes have to be calculated at compile time because the indexes are both
// used in device code and host code.
// The alternative would be to dynamically infer them at runtime and would come
// at an unnecessary cost on performance and code clarity.

template <class... RayTypesT>
class RayFactoryBase
{
    public:

    using RayTypes = std::tuple<RayTypesT...>;
    static constexpr unsigned int RayTypeCount = sizeof...(RayTypesT);

    template <typename RayT>
    RTAC_HOSTDEVICE
    static constexpr unsigned int index();

    // device side methods
    #ifdef RTAC_CUDACC
    template <class RayT> __device__
    static void trace(OptixTraversableHandle handle,
                      const float3& rayOrigin,
                      const float3& rayDirection,
                      float tmin, float tmax,
                      float rayTime,
                      OptixVisibilityMask mask,
                      unsigned int rayFlags,
                      //typename RayT::PayloadType& payload);
                      RayT& payload);
    #endif //RTAC_CUDACC
};

template <class... RayTypesT> template <typename RayT>
constexpr unsigned int RayFactoryBase<RayTypesT...>::index()
{
    static_assert(rtac::types::TypeInTuple<RayT, RayTypes>::value,
                  "RayT not registered within RayFactory. Cannot get index.");
    return rtac::types::TupleTypeIndex<RayT, RayTypes>::value;
}

#ifdef RTAC_CUDACC
template <class... RayTypesT> template <class RayT> __device__
void RayFactoryBase<RayTypesT...>::trace(OptixTraversableHandle handle,
                                         const float3& rayOrigin,
                                         const float3& rayDirection,
                                         float tmin, float tmax,
                                         float rayTime,
                                         OptixVisibilityMask mask,
                                         unsigned int rayFlags,
                                         //typename RayT::PayloadType& payload)
                                         RayT& payload)
{
    static_assert(rtac::types::TypeInTuple<RayT, RayTypes>::value,
                  "RayT not registered within RayFactory. Cannot trace.");

    // if constexpr is a c++17 feature which allows to ignore code at compile
    // time by checking the result of a constexpr test.
    if constexpr(RayT::PayloadType::RegisterCount == 0)
        optixTrace(handle, rayOrigin, rayDirection,
                   tmin, tmax, rayTime, mask, rayFlags,
                   index<RayT>(), RayTypeCount, index<RayT>());

    if constexpr(RayT::PayloadType::RegisterCount == 1)
        optixTrace(handle, rayOrigin, rayDirection,
                   tmin, tmax, rayTime, mask, rayFlags,
                   index<RayT>(), RayTypeCount, index<RayT>(),
                   payload.data()[0]);

    if constexpr(RayT::PayloadType::RegisterCount == 2)
        optixTrace(handle, rayOrigin, rayDirection,
                   tmin, tmax, rayTime, mask, rayFlags,
                   index<RayT>(), RayTypeCount, index<RayT>(),
                   payload.data()[0],
                   payload.data()[1]);

    if constexpr(RayT::PayloadType::RegisterCount == 3)
        optixTrace(handle, rayOrigin, rayDirection,
                   tmin, tmax, rayTime, mask, rayFlags,
                   index<RayT>(), RayTypeCount, index<RayT>(),
                   payload.data()[0],
                   payload.data()[1],
                   payload.data()[2]);

    if constexpr(RayT::PayloadType::RegisterCount == 4)
        optixTrace(handle, rayOrigin, rayDirection,
                   tmin, tmax, rayTime, mask, rayFlags,
                   index<RayT>(), RayTypeCount, index<RayT>(),
                   payload.data()[0],
                   payload.data()[1],
                   payload.data()[2],
                   payload.data()[3]);

    if constexpr(RayT::PayloadType::RegisterCount == 5)
        optixTrace(handle, rayOrigin, rayDirection,
                   tmin, tmax, rayTime, mask, rayFlags,
                   index<RayT>(), RayTypeCount, index<RayT>(),
                   payload.data()[0],
                   payload.data()[1],
                   payload.data()[2],
                   payload.data()[3],
                   payload.data()[4]);

    if constexpr(RayT::PayloadType::RegisterCount == 6)
        optixTrace(handle, rayOrigin, rayDirection,
                   tmin, tmax, rayTime, mask, rayFlags,
                   index<RayT>(), RayTypeCount, index<RayT>(),
                   payload.data()[0],
                   payload.data()[1],
                   payload.data()[2],
                   payload.data()[3],
                   payload.data()[4],
                   payload.data()[5]);

    if constexpr(RayT::PayloadType::RegisterCount == 7)
        optixTrace(handle, rayOrigin, rayDirection,
                   tmin, tmax, rayTime, mask, rayFlags,
                   index<RayT>(), RayTypeCount, index<RayT>(),
                   payload.data()[0],
                   payload.data()[1],
                   payload.data()[2],
                   payload.data()[3],
                   payload.data()[4],
                   payload.data()[5],
                   payload.data()[6]);

    if constexpr(RayT::PayloadType::RegisterCount == 8)
        optixTrace(handle, rayOrigin, rayDirection,
                   tmin, tmax, rayTime, mask, rayFlags,
                   index<RayT>(), RayTypeCount, index<RayT>(),
                   payload.data()[0],
                   payload.data()[1],
                   payload.data()[2],
                   payload.data()[3],
                   payload.data()[4],
                   payload.data()[5],
                   payload.data()[6],
                   payload.data()[7]);
}
#endif //RTAC_CUDACC

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_RAY_FACTORY_H_
