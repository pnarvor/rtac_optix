#ifndef _DEF_OPTIX_HELPERS_SAMPLES_ITEMS_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_ITEMS_H_

#include <array>

#include <optix_helpers/Context.h>
#include <optix_helpers/SceneItem.h>

#include <optix_helpers/samples/raytypes.h>
#include <optix_helpers/samples/models.h>
#include <optix_helpers/samples/materials.h>

namespace optix_helpers { namespace samples { namespace items {

SceneItem cube(const Context& context, const raytypes::RGB& rayType,
               float halfSize = 1.0, const std::array<float,3> color = {1.0,1.0,1.0});
SceneItem sphere(const Context& context, const raytypes::RGB& rayType,
                 float radius = 1.0, const std::array<float,3> color = {1.0,1.0,1.0});

}; //namespace items
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_ITEMS_H_
