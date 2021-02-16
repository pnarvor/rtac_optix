#ifndef _DEF_RTAC_OPTIX_MESH_ACCELERATION_STRUCT_H_
#define _DEF_RTAC_OPTIX_MESH_ACCELERATION_STRUCT_H_

#include <iostream>
#include <iomanip>
#include <cstring>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/DeviceMesh.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/AccelerationStruct.h>

namespace rtac { namespace optix {

class MeshAccelStruct : public AccelerationStruct
{
    public:

    using Ptr        = Handle<MeshAccelStruct>;
    using ConstPtr   = Handle<const MeshAccelStruct>;
    using DeviceMesh = rtac::cuda::DeviceMesh<float3, uint3>;
    template <typename T>
    using DeviceVector = rtac::cuda::DeviceVector<T>;

    static OptixBuildInput        default_build_input();
    static OptixAccelBuildOptions default_build_options();

    static Handle<DeviceMesh> cube_data(float scale = 1.0);

    protected:

    Handle<const DeviceMesh>  sourceMesh_;
    std::vector<CUdeviceptr> vertexBuffers_; // need to be in an array for motion blur.
    DeviceVector<float>       preTransform_; // Row-major homogeneous matrix without bottom line.
    std::vector<unsigned int> sbtFlags_;     // one per sbt record = material ?

    MeshAccelStruct(const Context::ConstPtr& context,
                    const Handle<const DeviceMesh>& mesh,
                    const DeviceVector<float>& preTransform = DeviceVector<float>(0),
                    const std::vector<unsigned int>& sbtFlags = std::vector<unsigned int>());

    public:

    static Ptr Create(const Context::ConstPtr& context,
                      const Handle<const DeviceMesh>& mesh,
                      const DeviceVector<float>& preTransform = DeviceVector<float>(0),
                      const std::vector<unsigned int>& sbtFlags = std::vector<unsigned int>());

    void set_mesh(const Handle<const DeviceMesh>& mesh);
    void set_pre_transform(const DeviceVector<float>& preTransform);
    void set_sbt_flags(const std::vector<unsigned int>& flags);
    void add_sbt_flags(unsigned int flag);

    void unset_pre_transform();
    void unset_sbt_flags();
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_MESH_ACCELERATION_STRUCT_H_
