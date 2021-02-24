#ifndef _DEF_RTAC_OPTIX_GEOMETRY_ACCEL_STRUCT_H_
#define _DEF_RTAC_OPTIX_GEOMETRY_ACCEL_STRUCT_H_

#include <iostream>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/AccelerationStruct.h>

namespace rtac { namespace optix {

class GeometryAccelStruct : public AccelerationStruct
{
    // This class is mostly there to provide a common base class to geometric
    // acceleration struct such as MeshAccelStruct and CustomAccelStruct.

    public:

    using Ptr      = Handle<GeometryAccelStruct>;
    using ConstPtr = Handle<const GeometryAccelStruct>;
    using Buffer   = AccelerationStruct::Buffer;

    static OptixBuildInput           default_build_input();
    static OptixAccelBuildOptions    default_build_options();
    static std::vector<unsigned int> default_geometry_flags();
    
    private:

    void update_sbt_flags();

    protected:

    std::vector<CUdeviceptr>  geomData_;
    std::vector<unsigned int> sbtFlags_;

    GeometryAccelStruct(const Context::ConstPtr& context,
                        const OptixBuildInput& buildInput = default_build_input(),
                        const OptixAccelBuildOptions& options = default_build_options());

    public:
    
    virtual void build(Buffer& tempBuffer, CUstream cudaStream = 0);

    void set_sbt_flags(const std::vector<unsigned int>& flags);
    void add_sbt_flags(unsigned int flag);
    void unset_sbt_flags();

    virtual unsigned int primitive_count() const = 0;

    virtual unsigned int sbt_width() const;
};

}; //namespace optix
}; //namespace rtac


#endif //_DEF_RTAC_OPTIX_GEOMETRY_ACCEL_STRUCT_H_
