#include <rtac_optix/GeometryAccelStruct.h>

namespace rtac { namespace optix {

OptixBuildInput GeometryAccelStruct::default_build_input()
{
    auto res = AccelerationStruct::default_build_input();
    return res;
}

OptixAccelBuildOptions GeometryAccelStruct::default_build_options()
{
    return AccelerationStruct::default_build_options();
}

std::vector<unsigned int> GeometryAccelStruct::default_geometry_flags()
{
    return std::vector<unsigned int>({OPTIX_GEOMETRY_FLAG_NONE});
}

GeometryAccelStruct::GeometryAccelStruct(const Context::ConstPtr& context,
                                         const OptixBuildInput& buildInput,
                                         const OptixAccelBuildOptions& options) :
        AccelerationStruct(context, buildInput, options)
{}

void GeometryAccelStruct::update_sbt_flags()
{
    switch(this->buildInput_.type)
    {
        case OPTIX_BUILD_INPUT_TYPE_TRIANGLES:
            if(sbtFlags_.size() > 0) {
                this->buildInput_.triangleArray.flags         = sbtFlags_.data();
                this->buildInput_.triangleArray.numSbtRecords = sbtFlags_.size();
            }
            else {
                this->buildInput_.triangleArray.flags         = nullptr;
                this->buildInput_.triangleArray.numSbtRecords = 0;
            }
            break;
        case OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES:
            if(sbtFlags_.size() > 0) {
                this->buildInput_.customPrimitiveArray.flags         = sbtFlags_.data();
                this->buildInput_.customPrimitiveArray.numSbtRecords = sbtFlags_.size();
            }
            else {
                this->buildInput_.customPrimitiveArray.flags         = nullptr;
                this->buildInput_.customPrimitiveArray.numSbtRecords = 0;
            }
            break;
        case OPTIX_BUILD_INPUT_TYPE_CURVES:
            throw std::logic_error("Curves not implemented yet.");
            break;
        case OPTIX_BUILD_INPUT_TYPE_INSTANCES:
        case OPTIX_BUILD_INPUT_TYPE_INSTANCE_POINTERS:
            throw std::runtime_error(
                "Invalid buildInput type (Instances) for GeometryAccelStruct");
            break;
        default:
            throw std::logic_error("Fatal error : Unknown build type. Check OptiX version");
            break;
    };
}

void GeometryAccelStruct::build(Buffer& tempBuffer, CUstream cudaStream)
{
    this->update_sbt_flags();
    AccelerationStruct::build(tempBuffer, cudaStream);
}

void GeometryAccelStruct::set_sbt_flags(const std::vector<unsigned int>& flags)
{
    sbtFlags_ = flags;
}

void GeometryAccelStruct::add_sbt_flags(unsigned int flag)
{
    sbtFlags_.push_back(flag);
}

void GeometryAccelStruct::unset_sbt_flags()
{
    sbtFlags_.clear();
}

unsigned int GeometryAccelStruct::sbt_width() const
{
    return sbtFlags_.size();
}

}; //namespace optix
}; //namespace rtac
