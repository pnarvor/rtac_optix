#ifndef _DEF_OPTIX_HELPERS_SAMPLES_MATERIALS_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_MATERIALS_H_

#include <optix_helpers/Context.h>
#include <optix_helpers/Material.h>
#include <optix_helpers/samples/raytypes.h>

namespace optix_helpers { namespace samples { namespace materials {

Material white(const Context& context, const raytypes::RGB& rayType);


}; //namespace materials
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_MATERIALS_H_


