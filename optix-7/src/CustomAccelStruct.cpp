#include <rtac_optix/CustomAccelStruct.h>

namespace rtac { namespace optix {

OptixBuildInput CustomAccelStruct::default_build_input()
{
    auto res = zero<OptixBuildInput>();
    res.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
    res.customPrimitiveArray.numPrimitives = 1; // locking this to 1 for now.
    return res;
}

OptixAccelBuildOptions CustomAccelStruct::default_build_options()
{
    return GeometryAccelStruct::default_build_options();
}

std::vector<unsigned int> CustomAccelStruct::default_geometry_flags()
{
    return std::vector<unsigned int>({OPTIX_GEOMETRY_FLAG_NONE});
}

CustomAccelStruct::CustomAccelStruct(const Context::ConstPtr& context) :
    GeometryAccelStruct(context, default_build_input(), default_build_options()),
    aabbBuffers_(1)
{
    this->set_aabb({-1,-1,-1,1,1,1});
    this->set_sbt_flags(default_geometry_flags());
}

CustomAccelStruct::Ptr CustomAccelStruct::Create(const Context::ConstPtr& context)
{
    return Ptr(new CustomAccelStruct(context));
}

void CustomAccelStruct::set_aabb(const std::vector<float>& aabb)
{
    aabb_ = aabb;
    aabbBuffers_[0] = reinterpret_cast<CUdeviceptr>(aabb_.data());
    this->buildInput_.customPrimitiveArray.aabbBuffers = aabbBuffers_.data();
}

void CustomAccelStruct::set_sbt_flags(const std::vector<unsigned int>& flags)
{
    sbtFlags_ = flags;
    this->buildInput_.customPrimitiveArray.flags = sbtFlags_.data();
    this->buildInput_.customPrimitiveArray.numSbtRecords = sbtFlags_.size();
}

void CustomAccelStruct::add_sbt_flags(unsigned int flag)
{
    sbtFlags_.push_back(flag);
    this->buildInput_.customPrimitiveArray.flags = sbtFlags_.data();
    this->buildInput_.customPrimitiveArray.numSbtRecords = sbtFlags_.size();
}

void CustomAccelStruct::unset_sbt_flags()
{
    sbtFlags_.clear();
    this->buildInput_.customPrimitiveArray.flags = nullptr;
    this->buildInput_.customPrimitiveArray.numSbtRecords = 0;
}

unsigned int CustomAccelStruct::primitive_count() const
{
    return this->buildInput_.customPrimitiveArray.numPrimitives;
}

}; //namespace optix
}; //namespace rtac
