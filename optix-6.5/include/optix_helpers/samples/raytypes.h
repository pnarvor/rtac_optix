#ifndef _DEF_OPTIX_HELPERS_SAMPLES_RAY_TYPES_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_RAY_TYPES_H_

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/Program.h>

namespace optix_helpers { namespace samples { namespace raytypes {

class RGB : public RayType
{
    public:

    static Index typeIndex;
    static const Source::ConstPtr typeDefinition;

    RGB(const Context::ConstPtr& context);

    static Program::Ptr rgb_miss_program(const Context::ConstPtr& context,
                                         const std::array<float,3>& color = {0,0,0});
    static Program::Ptr black_miss_program(const Context::ConstPtr& context);
};

}; //namespace raytypes
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_RAY_TYPES_H_


