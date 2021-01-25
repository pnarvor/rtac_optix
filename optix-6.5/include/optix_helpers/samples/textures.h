#ifndef _DEF_OPTIX_HELPERS_SAMPLES_TEXTURES_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_TEXTURES_H_

#include <array>

#include <optix_helpers/Context.h>
#include <optix_helpers/Buffer.h>
#include <optix_helpers/TextureSampler.h>

namespace optix_helpers { namespace samples { namespace textures {

TextureSampler::Ptr checkerboard(const Context::ConstPtr& context,
                                 const std::string& textureName = "checkerBoardTexture",
                                 const std::array<uint8_t,3>& color1 = {255,255,255},
                                 const std::array<uint8_t,3>& color2 = {0,0,0},
                                 size_t width = 64, size_t height = 64);

}; //namespace textures
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_TEXTURES_H_


