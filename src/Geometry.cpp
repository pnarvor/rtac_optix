#include <optix_helpers/Geometry.h>

namespace optix_helpers {

GeometryObj::GeometryObj(const optix::Geometry& geometry,
                         const Program& intersectionProgram,
                         const Program& boundingboxProgram) :
    geometry_(geometry),
    intersectionProgram_(intersectionProgram),
    boundingboxProgram_(boundingboxProgram)
{
}

optix::Geometry GeometryObj::geometry() const
{
    return geometry_;
}

Program GeometryObj::intersection_program() const
{
    return intersectionProgram_;
}

Program GeometryObj::boundingbox_program() const
{
    return boundingboxProgram_;
}

Geometry::Geometry() :
    Handle<GeometryObj>()
{}

Geometry::Geometry(const optix::Geometry& geometry,
                   const Program& intersectionProgram,
                   const Program& boundingboxProgram) :
    Handle<GeometryObj>(new GeometryObj(geometry, intersectionProgram, boundingboxProgram))
{}

Geometry::operator optix::Geometry() const
{
    return (*this)->geometry();
}

}; //namespace optix_helpers
