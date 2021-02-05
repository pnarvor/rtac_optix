#ifndef _DEF_RTAC_OPTIX_UTILS_H_
#define _DEF_RTAC_OPTIX_UTILS_H_

#include <iostream>
#include <cstring>
#include <sstream>

#include <cuda_runtime.h>
#include <optix.h>

#define OPTIX_CHECK( call )                                             \
    do {                                                                \
        OptixResult res = call;                                         \
        if(res != OPTIX_SUCCESS) {                                      \
            std::ostringstream oss;                                     \
            oss << "OptiX call '" << #call << "' failed '"              \
                << optixGetErrorName(res) << "' (code:" << res << ")\n" \
                << __FILE__ << ":" << __LINE__ << "\n";                 \
            throw std::runtime_error(oss.str());                        \
        }                                                               \
    } while(0)                                                          \


namespace rtac { namespace optix {

template <typename T>
T zero()
{
    // Simple helper to ensure a T struct is initialized to 0.
    // (helpful in an initializer list).
    T res;
    std::memset(&res, 0, sizeof(T));
    return res;
}

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_UTILS_H_
