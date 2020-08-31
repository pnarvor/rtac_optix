#ifndef _DEF_OPTIX_HELPERS_GEOMETRY_TRIANGLES_H_
#define _DEF_OPTIX_HELPERS_GEOMETRY_TRIANGLES_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Program.h>

namespace optix_helpers {

class GeometryTrianglesObj
{
    protected:

    optix::GeometryTriangles geometry_;

    public:

    GeometryTrianglesObj(const optix::GeometryTriangles& geometry);

    optix::GeometryTriangles geometry() const;
};

class GeometryTriangles : public Handle<GeometryTrianglesObj>
{
    public:

    GeometryTriangles();
    GeometryTriangles(const optix::GeometryTriangles& geometry);

    operator optix::GeometryTriangles() const;
};

}; //namespace optix_helpers
#endif //_DEF_OPTIX_HELPERS_GEOMETRY_TRIANGLES_H_
