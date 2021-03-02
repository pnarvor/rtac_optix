#ifndef _DEF_RTAC_OPTIX_GEOMETRY_ACCEL_STRUCT_H_
#define _DEF_RTAC_OPTIX_GEOMETRY_ACCEL_STRUCT_H_

#include <iostream>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_base/cuda/DeviceVector.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/AccelerationStruct.h>

namespace rtac { namespace optix {

class GeometryAccelStruct : public AccelerationStruct
{
    // This class is mostly there to provide a common base class to geometric
    // acceleration struct such as MeshAccelStruct and CustomAccelStruct.

    public:

    using Ptr                 = Handle<GeometryAccelStruct>;
    using ConstPtr            = Handle<const GeometryAccelStruct>;
    using Buffer              = AccelerationStruct::Buffer;
    using MaterialIndexBuffer = rtac::cuda::DeviceVector<uint8_t>;

    static OptixBuildInput           default_build_input();
    static OptixAccelBuildOptions    default_build_options();
    static std::vector<unsigned int> default_hit_flags();
    
    private:

    void update_hit_setup();

    protected:

    std::vector<CUdeviceptr>    geomData_;
    std::vector<unsigned int>   materialHitFlags_;
    Handle<MaterialIndexBuffer> materialIndexes_;


    GeometryAccelStruct(const Context::ConstPtr& context,
                        const OptixBuildInput& buildInput = default_build_input(),
                        const OptixAccelBuildOptions& options = default_build_options());

    public:
    
    virtual void build(Buffer& tempBuffer, CUstream cudaStream = 0);
    
    void material_hit_setup(const std::vector<unsigned int>& hitFlags,
                            const Handle<MaterialIndexBuffer>& materialIndexes = nullptr);
    void material_hit_setup(const std::vector<unsigned int>& hitFlags,
                            const std::vector<uint8_t>& materialIndexes);
    void clear_hit_setup();

    virtual unsigned int sbt_width() const;

    virtual unsigned int primitive_count() const = 0;
};

}; //namespace optix
}; //namespace rtac


#endif //_DEF_RTAC_OPTIX_GEOMETRY_ACCEL_STRUCT_H_
