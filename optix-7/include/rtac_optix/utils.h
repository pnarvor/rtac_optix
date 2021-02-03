#ifndef _DEF_RTAC_OPTIX_UTILS_H_
#define _DEF_RTAC_OPTIX_UTILS_H_

#include <iostream>
#include <sstream>

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

#endif //_DEF_RTAC_OPTIX_UTILS_H_
