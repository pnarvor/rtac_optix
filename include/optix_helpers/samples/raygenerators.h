#ifndef _DEF_OPTIX_HELPERS_SAMPLES_RAY_GENERATORS_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_RAY_GENERATORS_H_

#include <optix_helpers/Context.h>
#include <optix_helpers/PinHoleView.h>
#include <optix_helpers/RayGenerator.h>
#include <optix_helpers/samples/raytypes.h>

namespace optix_helpers { namespace samples { namespace raygenerators {

class RgbCamera : public RayGenerator<PinHoleView, RT_FORMAT_FLOAT3>
{
    public:

    using RayGeneratorType = RayGenerator<PinHoleView, RT_FORMAT_FLOAT3>;

    static const Source raygenSource;
    
    RgbCamera();
    RgbCamera(const Context& context, const RayType& rayType,
              size_t width, size_t height);
};

}; //namespace raygenerators
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_RAY_GENERATORS_H_
