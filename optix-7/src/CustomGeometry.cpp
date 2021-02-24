#include <rtac_optix/CustomGeometry.h>

namespace rtac { namespace optix {

OptixBuildInput CustomGeometry::default_build_input()
{
    auto res = zero<OptixBuildInput>();
    res.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    res.customPrimitiveArray.numPrimitives = 1; // locking this to 1 for now.
    return res;
}

OptixAccelBuildOptions CustomGeometry::default_build_options()
{
    return GeometryAccelStruct::default_build_options();
}

std::vector<unsigned int> CustomGeometry::default_geometry_flags()
{
    return std::vector<unsigned int>({OPTIX_GEOMETRY_FLAG_NONE});
}

CustomGeometry::CustomGeometry(const Context::ConstPtr& context) :
    GeometryAccelStruct(context, default_build_input(), default_build_options())
{
    this->set_aabb({-1,-1,-1,1,1,1});
    this->set_sbt_flags(default_geometry_flags());
}

CustomGeometry::Ptr CustomGeometry::Create(const Context::ConstPtr& context)
{
    return Ptr(new CustomGeometry(context));
}

void CustomGeometry::set_aabb(const std::vector<float>& aabb)
{
    aabb_ = aabb;
    if(this->geomData_.size() == 0)
        this->geomData_.resize(1);
    this->geomData_[0] = reinterpret_cast<CUdeviceptr>(aabb_.data());
    this->buildInput_.customPrimitiveArray.aabbBuffers = this->geomData_.data();
}

void CustomGeometry::set_sbt_flags(const std::vector<unsigned int>& flags)
{
    this->sbtFlags_ = flags;
    this->buildInput_.customPrimitiveArray.flags = this->sbtFlags_.data();
    this->buildInput_.customPrimitiveArray.numSbtRecords = this->sbtFlags_.size();
}

void CustomGeometry::add_sbt_flags(unsigned int flag)
{
    this->sbtFlags_.push_back(flag);
    this->buildInput_.customPrimitiveArray.flags = this->sbtFlags_.data();
    this->buildInput_.customPrimitiveArray.numSbtRecords = this->sbtFlags_.size();
}

void CustomGeometry::unset_sbt_flags()
{
    this->sbtFlags_.clear();
    this->buildInput_.customPrimitiveArray.flags = nullptr;
    this->buildInput_.customPrimitiveArray.numSbtRecords = 0;
}

unsigned int CustomGeometry::primitive_count() const
{
    return this->buildInput_.customPrimitiveArray.numPrimitives;
}

}; //namespace optix
}; //namespace rtac
