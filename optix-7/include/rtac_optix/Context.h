#ifndef _DEF_RTAC_OPTIX_CONTEXT_H_
#define _DEF_RTAC_OPTIX_CONTEXT_H_

#include <iostream>
#include <iomanip>
#include <cstring>

#include <optix.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>

namespace rtac { namespace optix {

class Context
{
    public:

    using ContextPtr = Handle<OptixDeviceContext>;

    protected:
    
    OptixDeviceContextOptions options_;
    mutable ContextPtr context_;

    static ContextPtr new_context(const OptixDeviceContextOptions& options,
                                  CUcontext cudaContext = 0);
    
    public:

    static OptixDeviceContextOptions default_options();
    static void log_callback(unsigned int level, const char* tag,
                             const char* message, void* data);

    Context();

    operator OptixDeviceContext() const;
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_CONTEXT_H_
