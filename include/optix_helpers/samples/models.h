#ifndef _DEF_OPTIX_HELPERS_SAMPLES_MODELS_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_MODELS_H_

#include <optix_helpers/Context.h>
#include <optix_helpers/Model.h>

namespace optix_helpers { namespace samples { namespace models {

Model cube(const Context& context, float halfSize = 1.0);
Model sphere(const Context& context, float radius = 1.0);

}; //namespace geometries
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_MODELS_H_
