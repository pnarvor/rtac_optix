#include <optix_helpers/Geometry.h>

namespace optix_helpers {

Geometry::Geometry(const optix::Geometry& geometry,
                   const Program& intersectionProgram,
                   const Program& boundingboxProgram,
                   size_t primitiveCount) :
    geometry_(geometry),
    intersectionProgram_(intersectionProgram),
    boundingboxProgram_(boundingboxProgram)
{
    geometry_->setIntersectionProgram(intersectionProgram_);
    geometry_->setBoundingBoxProgram(boundingboxProgram_);
    geometry_->setPrimitiveCount(primitiveCount);
}

Program Geometry::intersection_program() const
{
    return intersectionProgram_;
}

Program Geometry::boundingbox_program() const
{
    return boundingboxProgram_;
}

optix::Geometry Geometry::geometry() const
{
    return geometry_;
}

Geometry::operator optix::Geometry() const
{
    return geometry_;
}

optix::Geometry Geometry::operator->() const
{
    return geometry_;
}

}; //namespace optix_helpers
