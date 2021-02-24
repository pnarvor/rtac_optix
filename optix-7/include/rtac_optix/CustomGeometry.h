#ifndef _DEF_RTAC_OPTIX_CUSTOM_GEOMETRY_H_
#define _DEF_RTAC_OPTIX_CUSTOM_GEOMETRY_H_

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
#include <rtac_optix/GeometryAccelStruct.h>

namespace rtac { namespace optix {

class CustomGeometry : public GeometryAccelStruct
{
    public:

    using Ptr      = Handle<CustomGeometry>;
    using ConstPtr = Handle<const CustomGeometry>;
    template <typename T>
    using DeviceVector = rtac::cuda::DeviceVector<T>;

    static OptixBuildInput        default_build_input();
    static OptixAccelBuildOptions default_build_options();
    static std::vector<unsigned int> default_geometry_flags();

    protected:
    
    // Axis Aligned Bounding Box.
    DeviceVector<float> aabb_;

    CustomGeometry(const Context::ConstPtr& context);

    public:

    static Ptr Create(const Context::ConstPtr& context);

    void set_aabb(const std::vector<float>& aabb);

    virtual void set_sbt_flags(const std::vector<unsigned int>& flags);
    virtual void add_sbt_flags(unsigned int flag);
    virtual void unset_sbt_flags();

    virtual unsigned int primitive_count() const;
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_CUSTOM_GEOMETRY_H_
