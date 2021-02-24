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

}; //namespace optix
}; //namespace rtac
