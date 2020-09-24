#include <optix_helpers/utils.h>

namespace optix_helpers {

inline optix::float4 make_float4(const rtac::types::Rectangle<float>& rect)
{
    return optix::make_float4(rect.left, rect.right, rect.bottom, rect.top);
}

}; //namespace optix_helpers

