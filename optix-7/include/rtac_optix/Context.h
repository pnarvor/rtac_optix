#ifndef _DEF_RTAC_OPTIX_CONTEXT_H_
#define _DEF_RTAC_OPTIX_CONTEXT_H_

#include <iostream>
#include <iomanip>
#include <cstring>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>

namespace rtac { namespace optix {

class Context
{
    public:

    using Ptr      = Handle<Context>;
    using ConstPtr = Handle<const Context>;

    static OptixDeviceContextOptions default_options();
    static void log_callback(unsigned int level, const char* tag,
                             const char* message, void* data);

    protected:

    mutable OptixDeviceContext context_;

    // The Context class may be shared across several entities.  Making the
    // constructor protected and deleting the copy constructor and copy
    // assignment allows to force the user to created a shared pointer through
    // the Create static method to get a new context.
    Context(const OptixDeviceContext& context);

    public:

    Context(const Context& other)            = delete;
    Context& operator=(const Context& other) = delete;

    static Ptr Create(const OptixDeviceContextOptions& options = default_options(),
                      CUcontext cudaContext = 0,
                      bool diskCacheEnabled = false);
    ~Context();

    // Implicitly castable to OptixDeviceContext for seamless use in optix API.
    // This breaks encapsulation.
    // /!\ Use only in optix API calls except for optixDeviceContextDestroy,
    operator OptixDeviceContext() const; 

    void enable_cache();
    void disable_cache();
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_CONTEXT_H_
