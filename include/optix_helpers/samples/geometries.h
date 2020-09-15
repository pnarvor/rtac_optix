#ifndef _DEF_OPTIX_HELPERS_SAMPLES_GEOMETRIES_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_GEOMETRIES_H_

#include <optix_helpers/Context.h>
#include <optix_helpers/Program.h>
#include <optix_helpers/Geometry.h>
#include <optix_helpers/GeometryTriangles.h>

namespace optix_helpers { namespace samples { namespace geometries {

GeometryTriangles cube(const Context& context, float scale = 1.0);
Geometry          sphere(const Context& context, float radius = 1.0);
GeometryTriangles square(const Context& context, float scale = 1.0);

GeometryTriangles indexed_cube(const Context& context, float scale = 1.0);



}; //namespace geometries
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_GEOMETRIES_H_

