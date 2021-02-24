#ifndef _RTAC_OPTIX_TRAVERSABLE_HANDLE_H_
#define _RTAC_OPTIX_TRAVERSABLE_HANDLE_H_

#include <iostream>
#include <iomanip>
#include <cstring>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_base/cuda/utils.h>
#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/TraversableHandle.h>

namespace rtac { namespace optix {

class AccelerationStruct : public TraversableHandle
{
    // ABSTRACT class

    // Represent complex nodes of the rendering graph which need to be built.
    // The purpose of this graph is to hide the build operation. It must be
    // subclassed by a class implementing the sbt_width method.

    public:

    using Ptr      = Handle<AccelerationStruct>;
    using ConstPtr = Handle<const AccelerationStruct>;

    using Buffer = rtac::cuda::DeviceVector<unsigned char>;

    static OptixBuildInput        default_build_input();
    static OptixAccelBuildOptions default_build_options();

    protected:
    
    Context::ConstPtr      context_;
    OptixTraversableHandle handle_;
    OptixBuildInput        buildInput_;
    OptixAccelBuildOptions buildOptions_;
    Buffer                 buffer_; // contains data after build

    AccelerationStruct(const Context::ConstPtr& context,
                       const OptixBuildInput& buildInput = default_build_input(),
                       const OptixAccelBuildOptions& buildOptions = default_build_options());

    public:

    //static Ptr Create(const Context::ConstPtr& context,
    //                   const OptixBuildInput& buildInput = default_build_input(),
    //                   const OptixAccelBuildOptions& buildOptions = default_build_options());

    virtual void build(Buffer& tempBuffer, CUstream cudaStream = 0);
    void build(CUstream cudaStream = 0);
    
    CUdeviceptr data();

    OptixBuildInput& build_input();
    const OptixBuildInput& build_input() const;

    virtual operator OptixTraversableHandle();
    virtual unsigned int sbt_width() const = 0;
};

}; //namespace optix
}; //namespace rtac

#endif //_RTAC_OPTIX_TRAVERSABLE_HANDLE_H_
