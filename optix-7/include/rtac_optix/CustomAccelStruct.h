#ifndef _DEF_RTAC_OPTIX_CUSTOM_ACCEL_STRUCT_H_
#define _DEF_RTAC_OPTIX_CUSTOM_ACCEL_STRUCT_H_

#include <iostream>
#include <iomanip>
#include <cstring>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_base/cuda/DeviceVector.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/AccelerationStruct.h>

namespace rtac { namespace optix {

class CustomAccelStruct : public AccelerationStruct
{
    public:

    using Ptr      = Handle<CustomAccelStruct>;
    using ConstPtr = Handle<const CustomAccelStruct>;
    template <typename T>
    using DeviceVector = rtac::cuda::DeviceVector<T>;

    static OptixBuildInput        default_build_input();
    static OptixAccelBuildOptions default_build_options();
    static std::vector<unsigned int> default_geometry_flags();

    protected:
    
    // Axis Aligned Bounding Box.
    DeviceVector<float>       aabb_;
    std::vector<CUdeviceptr>  aabbBuffers_; // need to be in an array for motion blur.
    std::vector<unsigned int> sbtFlags_;   // one per sbt record = material ?

    CustomAccelStruct(const Context::ConstPtr& context);

    public:

    static Ptr Create(const Context::ConstPtr& context);

    void set_aabb(const std::vector<float>& aabb);

    void set_sbt_flags(const std::vector<unsigned int>& flags);
    void add_sbt_flags(unsigned int flag);
    void unset_sbt_flags();
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_CUSTOM_ACCEL_STRUCT_H_
