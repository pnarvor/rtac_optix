#include <optix_helpers/Geometry.h>

namespace optix_helpers {

GeometryObj::GeometryObj(const optix::Geometry& geometry,
                   const Program& intersectionProgram,
                   const Program& boundingboxProgram,
                   size_t primitiveCount) :
    geometry_(geometry),
    intersectionProgram_(intersectionProgram),
    boundingboxProgram_(boundingboxProgram)
{
    geometry_->setIntersectionProgram(*intersectionProgram_);
    geometry_->setBoundingBoxProgram(*boundingboxProgram_);
    geometry_->setPrimitiveCount(primitiveCount);
}

Program GeometryObj::intersection_program() const
{
    return intersectionProgram_;
}

Program GeometryObj::boundingbox_program() const
{
    return boundingboxProgram_;
}

optix::Geometry GeometryObj::geometry() const
{
    return geometry_;
}

GeometryObj::operator optix::Geometry() const
{
    return geometry_;
}

optix::Geometry GeometryObj::operator->() const
{
    return geometry_;
}

}; //namespace optix_helpers
