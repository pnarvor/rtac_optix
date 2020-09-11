#ifndef _DEF_OPTIX_HELPERS_UTILS_H_
#define _DEF_OPTIX_HELPERS_UTILS_H_

#include <iostream>
#include <array>

#include <optixu/optixpp.h>

#include <rtac_base/types/common.h>

namespace optix_helpers {

template<typename Derived>
inline optix::float3 make_float3(const Eigen::DenseBase<Derived>& v)
{
    return optix::make_float3(v(0), v(1), v(2));
}

template<typename T>
inline optix::float3 make_float3(const std::array<T,3>& v)
{
    return optix::make_float3(v[0], v[1], v[2]);
}

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_UTILS_H_
