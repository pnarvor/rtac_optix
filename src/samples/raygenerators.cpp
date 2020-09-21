#include <optix_helpers/samples/raygenerators.h>

namespace optix_helpers { namespace samples { namespace raygenerators {

const Source RgbCamera::raygenSource = Source(R"(
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
    
RgbCamera::RgbCamera()
{}

RgbCamera::RgbCamera(const Context& context, const RayType& rayType,
                     size_t width, size_t height) :
    RayGeneratorType(new RayGeneratorObj<PinHoleView, RT_FORMAT_FLOAT3>(
                 context, rayType, RgbCamera::raygenSource, "renderBuffer"))
{
    (*this)->set_size(width, height);
}

}; //namespace raygenerators
}; //namespace samples
}; //namespace optix_helpers

