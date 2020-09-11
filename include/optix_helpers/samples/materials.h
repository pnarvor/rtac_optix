#ifndef _DEF_OPTIX_HELPERS_SAMPLES_MATERIALS_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_MATERIALS_H_

#include <array>

#include <optix_helpers/utils.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/Material.h>
#include <optix_helpers/samples/raytypes.h>

namespace optix_helpers { namespace samples { namespace materials {

Material rgb(const Context& context, const raytypes::RGB& rayType, 
             const std::array<float,3>& color = {1.0,1.0,1.0});
Material white(const Context& context, const raytypes::RGB& rayType);
Material black(const Context& context, const raytypes::RGB& rayType);
Material red(const Context& context, const raytypes::RGB& rayType);
Material green(const Context& context, const raytypes::RGB& rayType);
Material blue(const Context& context, const raytypes::RGB& rayType);


}; //namespace materials
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_MATERIALS_H_


