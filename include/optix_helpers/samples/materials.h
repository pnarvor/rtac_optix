#ifndef _DEF_OPTIX_HELPERS_SAMPLES_MATERIALS_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_MATERIALS_H_

#include <array>

#include <optix_helpers/utils.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/Material.h>
#include <optix_helpers/TextureSampler.h>
#include <optix_helpers/TexturedMaterial.h>
#include <optix_helpers/samples/raytypes.h>
#include <optix_helpers/samples/textures.h>

namespace optix_helpers { namespace samples { namespace materials {

Material rgb(const Context& context, const raytypes::RGB& rayType, 
             const std::array<float,3>& color = {1.0,1.0,1.0});
Material white(const Context& context, const raytypes::RGB& rayType);
Material black(const Context& context, const raytypes::RGB& rayType);
Material red(const Context& context, const raytypes::RGB& rayType);
Material green(const Context& context, const raytypes::RGB& rayType);
Material blue(const Context& context, const raytypes::RGB& rayType);

TexturedMaterial checkerboard(const Context& context, const raytypes::RGB& rayType,
                              const std::array<uint8_t,3>& color1 = {255,255,255},
                              const std::array<uint8_t,3>& color2 = {0,0,0},
                              size_t width = 64, size_t height = 64);


}; //namespace materials
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_MATERIALS_H_


