#include <rtac_optix/MeshAccelStruct.h>

namespace rtac { namespace optix {

OptixBuildInput MeshAccelStruct::default_build_input()
{
    auto res = GeometryAccelStruct::default_build_input();
    res.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    return res;
}

OptixAccelBuildOptions MeshAccelStruct::default_build_options()
{
    return GeometryAccelStruct::default_build_options();
}

std::vector<unsigned int> MeshAccelStruct::default_geometry_flags()
{
    return std::vector<unsigned int>({OPTIX_GEOMETRY_FLAG_NONE});
}

Handle<MeshAccelStruct::DeviceMesh> MeshAccelStruct::cube_data(float scale)
{
    return Handle<DeviceMesh>(new DeviceMesh(rtac::types::Mesh<float3,uint3>::cube(scale)));
}

MeshAccelStruct::MeshAccelStruct(const Context::ConstPtr& context,
                                 const Handle<const DeviceMesh>& mesh,
                                 const DeviceVector<float>& preTransform,
                                 const std::vector<unsigned int>& sbtFlags) :
    GeometryAccelStruct(context, default_build_input(), default_build_options()),
    sourceMesh_(NULL)
{
    this->set_mesh(mesh);
    this->set_pre_transform(preTransform);
    if(sbtFlags.size() > 0)
        this->set_sbt_flags(sbtFlags);
    else
        this->set_sbt_flags(default_geometry_flags());
}

MeshAccelStruct::Ptr MeshAccelStruct::Create(const Context::ConstPtr& context,
                                             const Handle<const DeviceMesh>& mesh,
                                             const DeviceVector<float>& preTransform,
                                             const std::vector<unsigned int>& sbtFlags)
{
    return Ptr(new MeshAccelStruct(context, mesh, preTransform, sbtFlags));
}

void MeshAccelStruct::set_mesh(const Handle<const DeviceMesh>& mesh)
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

void MeshAccelStruct::set_pre_transform(const DeviceVector<float>& preTransform)
{
    if(preTransform.size() == 0) {
        this->unset_pre_transform();
        return;
    }
    if(preTransform.size() != 12) {
        std::ostringstream oss;
        oss << "MeshAccelStruct : preTransform mush be a 12 sized vector "
            << "(3 first rows of a row major homogeneous matrix).";
        throw std::runtime_error(oss.str());
    }
    preTransform_ = preTransform;
    this->buildInput_.triangleArray.preTransform = 
        reinterpret_cast<CUdeviceptr>(preTransform_.data());
    this->buildInput_.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_MATRIX_FLOAT12;
}

void MeshAccelStruct::unset_pre_transform()
{
    this->buildInput_.triangleArray.preTransform    = 0;
    this->buildInput_.triangleArray.transformFormat = OPTIX_TRANSFORM_FORMAT_NONE;
}

void MeshAccelStruct::set_sbt_flags(const std::vector<unsigned int>& flags)
{
    sbtFlags_ = flags;
    this->buildInput_.triangleArray.flags = sbtFlags_.data();
    this->buildInput_.triangleArray.numSbtRecords = sbtFlags_.size();
}

void MeshAccelStruct::add_sbt_flags(unsigned int flag)
{
    sbtFlags_.push_back(flag);
    this->buildInput_.triangleArray.flags = sbtFlags_.data();
    this->buildInput_.triangleArray.numSbtRecords = sbtFlags_.size();
}

void MeshAccelStruct::unset_sbt_flags()
{
    sbtFlags_.clear();
    this->buildInput_.triangleArray.flags = nullptr;
    this->buildInput_.triangleArray.numSbtRecords = 0;
}

unsigned int MeshAccelStruct::primitive_count() const
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
