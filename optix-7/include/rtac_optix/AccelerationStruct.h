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

namespace rtac { namespace optix {

class AccelerationStruct
{
    // This class represent a node of the scene graph in which the ray will
    // propagate. OptixTraversableHandle is a generic type which can represent
    // any item in the graph (geometries, transforms, "structural" nodes
    // containing other nodes, etc...)
    
    public:

    using Ptr      = Handle<AccelerationStruct>;
    using ConstPtr = Handle<const AccelerationStruct>;

    using Buffer = rtac::cuda::DeviceVector<unsigned char>;

    protected:
    
    Context::ConstPtr      context_;
    OptixTraversableHandle handle_;
    Buffer                 buffer_; // contains data after build

    AccelerationStruct(const Context::ConstPtr& context);

    public:

    static Ptr Create(const Context::ConstPtr& context);

    void build(const OptixBuildInput& buildInput,
               const OptixAccelBuildOptions& buildOptions,
               Buffer& tempBuffer, CUstream cudaStream = 0);
    void build(const OptixBuildInput& buildInput,
               const OptixAccelBuildOptions& buildOptions,
               CUstream cudaStream = 0);
    
    // Implicitly castable to OptixPipeline for seamless use in optix API.
    // This breaks encapsulation.
    // /!\ Use only in optix API calls except for optixDeviceContextDestroy,
    operator OptixTraversableHandle() const;
    CUdeviceptr data();
};

}; //namespace optix
}; //namespace rtac

#endif //_RTAC_OPTIX_TRAVERSABLE_HANDLE_H_
