#include <optix_helpers/GeometryTriangles.h>

namespace optix_helpers {

GeometryTrianglesObj::GeometryTrianglesObj(const optix::GeometryTriangles& geometry) :
    geometry_(geometry)
{}

optix::GeometryTriangles GeometryTrianglesObj::geometry() const
{
    return geometry_;
}

GeometryTriangles::GeometryTriangles() :
    Handle<GeometryTrianglesObj>()
{}

GeometryTriangles::GeometryTriangles(const optix::GeometryTriangles& geometry) :
    Handle<GeometryTrianglesObj>(new GeometryTrianglesObj(geometry))
{}

GeometryTriangles::operator optix::GeometryTriangles() const
{
    return (*this)->geometry();
}

}; //namespace optix_helpers

