#include <rtac_optix/MeshGeometry.h>

namespace rtac { namespace optix {

OptixBuildInput MeshGeometry::default_build_input()
{
    auto res = GeometryAccelStruct::default_build_input();
    res.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    return res;
}

OptixAccelBuildOptions MeshGeometry::default_build_options()
{
    return GeometryAccelStruct::default_build_options();
}

Handle<MeshGeometry::DeviceMesh> MeshGeometry::cube_data(float scale)
{
    return Handle<DeviceMesh>(new DeviceMesh(rtac::types::Mesh<float3,uint3>::cube(scale)));
}

MeshGeometry::MeshGeometry(const Context::ConstPtr& context,
                           const Handle<const DeviceMesh>& mesh,
                           const DeviceVector<float>& preTransform) :
    GeometryAccelStruct(context, default_build_input(), default_build_options()),
    sourceMesh_(NULL)
{
    this->set_mesh(mesh);
    this->set_pre_transform(preTransform);
}

MeshGeometry::Ptr MeshGeometry::Create(const Context::ConstPtr& context,
                                       const Handle<const DeviceMesh>& mesh,
                                       const DeviceVector<float>& preTransform)
{
    return Ptr(new MeshGeometry(context, mesh, preTransform));
}

MeshGeometry::Ptr MeshGeometry::CreateCube(const Context::ConstPtr& context,
                                           float scale,
                                           const DeviceVector<float>& preTransform)
{
    return Create(context, cube_data(scale), preTransform);
}

void MeshGeometry::set_mesh(const Handle<const DeviceMesh>& mesh)
{
    if(mesh->num_points() == 0)
        return;

    sourceMesh_ = mesh; // Keeping a reference to mesh to keep it alive.
                        // Can be released after build.

    // Keeping vertex data pointer in a vector is mandatory because the buildInput
    // expect an array of vertex buffers for motion blur calculations.
    if(this->geomData_.size() == 0)
        this->geomData_.resize(1);
    this->geomData_[0] = reinterpret_cast<CUdeviceptr>(mesh->points().data());

    this->buildInput_.triangleArray.numVertices         = mesh->num_points();
    this->buildInput_.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    this->buildInput_.triangleArray.vertexStrideInBytes = sizeof(DeviceMesh::Point);
    this->buildInput_.triangleArray.vertexBuffers       = geomData_.data();
        

    if(mesh->num_faces() > 0) {
        this->buildInput_.triangleArray.numIndexTriplets    = mesh->num_faces();
        this->buildInput_.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        this->buildInput_.triangleArray.indexStrideInBytes  = sizeof(DeviceMesh::Face);
        this->buildInput_.triangleArray.indexBuffer         =
            reinterpret_cast<CUdeviceptr>(mesh->faces().data());
    }
    else {
        // Erasing possible previously defined index faces.
        this->buildInput_.triangleArray.numIndexTriplets    = 0;
        this->buildInput_.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_NONE;
        this->buildInput_.triangleArray.indexStrideInBytes  = 0;
        this->buildInput_.triangleArray.indexBuffer         = 0;
    }
}

void MeshGeometry::set_pre_transform(const DeviceVector<float>& preTransform)
{
    if(preTransform.size() == 0) {
        this->unset_pre_transform();
        return;
    }
    if(preTransform.size() != 12) {
        std::ostringstream oss;
        oss << "MeshGeometry : preTransform mush be a 12 sized vector "
            << "(3 first rows of a row major homogeneous matrix).";
        throw std::runtime_error(oss.str());
    }
    preTransform_ = preTransform;
    this->buildInput_.triangleArray.preTransform = 
        reinterpret_cast<CUdeviceptr>(preTransform_.data());
    this->buildInput_.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
}

void MeshGeometry::unset_pre_transform()
{
    this->buildInput_.triangleArray.preTransform    = 0;
    this->buildInput_.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;
}

void MeshGeometry::enable_vertex_access()
{
    this->buildOptions_.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
}

void MeshGeometry::disable_vertex_access()
{
    this->buildOptions_.buildFlags &= ~OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
}

unsigned int MeshGeometry::primitive_count() const
{
    // If we have an index buffer set, returning its size. If the index buffer
    // is empty, triangles vertices are assumed to be packed into
    // vertex buffers.
    if(this->buildInput_.triangleArray.numIndexTriplets != 0) {
        return this->buildInput_.triangleArray.numIndexTriplets;
    }
    else {
        return this->buildInput_.triangleArray.numVertices / 3;
    }
}

}; //namespace optix
}; //namespace rtac
