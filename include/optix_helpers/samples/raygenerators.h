#ifndef _DEF_OPTIX_HELPERS_SAMPLES_RAY_GENERATORS_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_RAY_GENERATORS_H_

#include <optix_helpers/Context.h>
#include <optix_helpers/PinHoleView.h>
#include <optix_helpers/RayGenerator.h>
#include <optix_helpers/samples/raytypes.h>

namespace optix_helpers { namespace samples { namespace raygenerators {

template <typename RenderBufferType>
class RgbCamera : public RayGenerator<PinHoleView, RT_FORMAT_FLOAT3, RenderBufferType>
{
    public:

    using RayGeneratorType = RayGenerator<PinHoleView, RT_FORMAT_FLOAT3, RenderBufferType>;

    static const Source raygenSource;
    
    RgbCamera();
    RgbCamera(const Context& context, const RayType& rayType,
              size_t width, size_t height);
};

template <typename RenderBufferType>
RgbCamera<RenderBufferType>::RgbCamera()
{}

template <typename RenderBufferType>
RgbCamera<RenderBufferType>::RgbCamera(const Context& context, const RayType& rayType,
                     size_t width, size_t height) :
    RayGeneratorType(context, rayType, RgbCamera::raygenSource, "renderBuffer")
{
    (*this)->set_size(width, height);
}

template <typename RenderBufferType>
const Source RgbCamera<RenderBufferType>::raygenSource = Source(R"(
#include <optix.h>

using namespace optix;

#include <rays/RGB.h>
#include <view/pinhole.h>

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);
rtDeclareVariable(rtObject, topObject,,);

rtBuffer<float3, 2> renderBuffer;

RT_PROGRAM void rgb_camera()
{
    raytypes::RGB payload;
    payload.color = make_float3(0.0f,0.0f,0.0f);

    Ray ray = pinhole_ray(launchIndex, 0);

    rtTrace(topObject, ray, payload);
    renderBuffer[launchIndex] = payload.color;
}
)", "rgb_camera");
    


}; //namespace raygenerators
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_RAY_GENERATORS_H_
