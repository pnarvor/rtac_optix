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

    using Buffer = rtac::cuda::DeviceVector<unsigned char>;

    protected:
    
    Context::ConstPtr      context_;
    OptixTraversableHandle handle_;
    Handle<Buffer>         buffer_; // contains data after build

    public:

    AccelerationStruct(const Context::ConstPtr& context);

    void build(const OptixBuildInput& buildInput,
               const OptixAccelBuildOptions& buildOptions,
               Buffer& tempBuffer, CUstream cudaStream = 0);
    void build(const OptixBuildInput& buildInput,
               const OptixAccelBuildOptions& buildOptions,
               CUstream cudaStream = 0);
    
    CUdeviceptr data();
    operator OptixTraversableHandle();
};

}; //namespace optix
}; //namespace rtac

#endif //_RTAC_OPTIX_TRAVERSABLE_HANDLE_H_
