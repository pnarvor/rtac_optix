#ifndef _DEF_RTAC_OPTIX_INSTANCE_ACCEL_STRUCT_H_
#define _DEF_RTAC_OPTIX_INSTANCE_ACCEL_STRUCT_H_

#include <iostream>
#include <cstring>
#include <vector>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_base/cuda/HostVector.h>
#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/DeviceMesh.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/AccelerationStruct.h>
#include <rtac_optix/Instance.h>

namespace rtac { namespace optix {

class InstanceAccelStruct : public AccelerationStruct
{
    public:

    using Ptr      = Handle<InstanceAccelStruct>;
    using ConstPtr = Handle<const InstanceAccelStruct>;

    using Buffer    = AccelerationStruct::Buffer;
    using Instances = std::vector<Instance::Ptr>;

    static OptixBuildInput        default_build_input();
    static OptixAccelBuildOptions default_build_options();

    protected:

    Instances instances_;

    // tmpInstanceData_ is used only for the build operation, but must stay in memory
    // because the build operation is asynchronous.
    cuda::DeviceVector<OptixInstance> tmpInstanceData_; 

    InstanceAccelStruct(const Context::ConstPtr& context);

    public:

    static Ptr Create(const Context::ConstPtr& context);
    virtual void build(Buffer& tempBuffer, CUstream cudaStream = 0);

    void add_instance(const Instance::Ptr& instance);

    virtual unsigned int sbt_width() const;
};

}; //namespace optix
}; //namespace rtac


#endif //_DEF_RTAC_OPTIX_INSTANCE_ACCEL_STRUCT_H_
