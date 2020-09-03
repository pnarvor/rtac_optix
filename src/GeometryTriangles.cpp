#include <optix_helpers/GeometryTriangles.h>

namespace optix_helpers {

GeometryTrianglesObj::GeometryTrianglesObj(const optix::GeometryTriangles& geometry,
                                           const optix::Buffer& points,
                                           const optix::Buffer& faces) :
    geometry_(geometry),
    points_(points),
    faces_(faces)
{}

optix::GeometryTriangles GeometryTrianglesObj::geometry() const
{
    return geometry_;
}

optix::Buffer GeometryTrianglesObj::points() const
{
    return points_;
}

optix::Buffer GeometryTrianglesObj::faces() const
{
    return faces_;
}

GeometryTriangles::GeometryTriangles() :
    Handle<GeometryTrianglesObj>()
{}

GeometryTriangles::GeometryTriangles(const optix::GeometryTriangles& geometry,
                                     const optix::Buffer& points,
                                     const optix::Buffer& faces) :
    Handle<GeometryTrianglesObj>(new GeometryTrianglesObj(geometry, points, faces))
{}

GeometryTriangles::operator optix::GeometryTriangles() const
{
    return (*this)->geometry();
}

}; //namespace optix_helpers

