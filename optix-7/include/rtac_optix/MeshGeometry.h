#ifndef _DEF_RTAC_OPTIX_MESH_GEOMETRY_H_
#define _DEF_RTAC_OPTIX_MESH_GEOMETRY_H_

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
#include <rtac_optix/GeometryAccelStruct.h>

namespace rtac { namespace optix {

class MeshGeometry : public GeometryAccelStruct
{
    public:

    using Ptr        = Handle<MeshGeometry>;
    using ConstPtr   = Handle<const MeshGeometry>;
    using DeviceMesh = rtac::cuda::DeviceMesh<float3, uint3>;
    template <typename T>
    using DeviceVector = rtac::cuda::DeviceVector<T>;

    static OptixBuildInput           default_build_input();
    static OptixAccelBuildOptions    default_build_options();

    static Handle<DeviceMesh> cube_data(float scale = 1.0);

    protected:

    Handle<const DeviceMesh>  sourceMesh_;
    DeviceVector<float>       preTransform_;  // Row-major homogeneous matrix without bottom line.

    MeshGeometry(const Context::ConstPtr& context,
                 const Handle<const DeviceMesh>& mesh,
                 const DeviceVector<float>& preTransform = DeviceVector<float>(0));

    public:

    static Ptr Create(const Context::ConstPtr& context,
                      const Handle<const DeviceMesh>& mesh,
                      const DeviceVector<float>& preTransform = DeviceVector<float>(0));

    void set_mesh(const Handle<const DeviceMesh>& mesh);

    void set_pre_transform(const DeviceVector<float>& preTransform);
    void unset_pre_transform();

    virtual unsigned int primitive_count() const;
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_MESH_GEOMETRY_H_
