#include <optix_helpers/GeometryTriangles.h>

namespace optix_helpers {

GeometryTrianglesObj::GeometryTrianglesObj(const optix::GeometryTriangles& geometry,
                                           const optix::Buffer& points,
                                           const optix::Buffer& faces) :
    geometry_(geometry),
    points_(points),
    faces_(faces)
{}

optix::Buffer GeometryTrianglesObj::points() const
{
    return points_;
}

optix::Buffer GeometryTrianglesObj::faces() const
{
    return faces_;
}

optix::GeometryTriangles GeometryTrianglesObj::geometry() const
{
    return geometry_;
}

GeometryTrianglesObj::operator optix::GeometryTriangles() const
{
    return geometry_;
}

optix::GeometryTriangles GeometryTrianglesObj::operator->() const
{
    return geometry_;
}

}; //namespace optix_helpers

