#ifndef _DEF_OPTIX_HELPERS_SAMPLES_MATERIALS_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_MATERIALS_H_

#include <array>

#include <optix_helpers/utils.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/Material.h>
#include <optix_helpers/TextureSampler.h>
#include <optix_helpers/TexturedMaterial.h>
#include <optix_helpers/samples/maths.h>
#include <optix_helpers/samples/raytypes.h>
#include <optix_helpers/samples/textures.h>

namespace optix_helpers { namespace samples { namespace materials {

Material::Ptr rgb(const Context::ConstPtr& context, const raytypes::RGB& rayType, 
             const std::array<float,3>& color = {1.0,1.0,1.0});
Material::Ptr white(const Context::ConstPtr& context, const raytypes::RGB& rayType);
Material::Ptr black(const Context::ConstPtr& context, const raytypes::RGB& rayType);
Material::Ptr red(const Context::ConstPtr& context, const raytypes::RGB& rayType);
Material::Ptr green(const Context::ConstPtr& context, const raytypes::RGB& rayType);
Material::Ptr blue(const Context::ConstPtr& context, const raytypes::RGB& rayType);

Material::Ptr lambert(const Context::ConstPtr& context, const raytypes::RGB& raytype,
                 const std::array<float,3>& light = {0.0,0.0,0.0},
                 const std::array<float,3>& color = {1.0,1.0,1.0});

Material::Ptr barycentrics(const Context::ConstPtr& context, const raytypes::RGB& rayType);

TexturedMaterial::Ptr checkerboard(const Context::ConstPtr& context,
                                   const raytypes::RGB& rayType,
                                   const std::array<uint8_t,3>& color1 = {255,255,255},
                                   const std::array<uint8_t,3>& color2 = {0,0,0},
                                   size_t width = 64, size_t height = 64);

Material::Ptr perfect_mirror(const Context::ConstPtr& context, const raytypes::RGB& rayType);
Material::Ptr perfect_refraction(const Context::ConstPtr& context, const raytypes::RGB& rayType,
                            float refractiveIndex = 2.417);


}; //namespace materials
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_MATERIALS_H_


