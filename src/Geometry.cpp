#include <optix_helpers/Geometry.h>

namespace optix_helpers {

Geometry::Ptr Geometry::New(const Context::ConstPtr& context,
                            const Program::Ptr& intersectionProgram,
                            const Program::Ptr& boundingboxProgram,
                            size_t primitiveCount)
{
    return Ptr(new Geometry(context,
                            intersectionProgram,
                            boundingboxProgram,
                            primitiveCount));
}

Geometry::Geometry(const Context::ConstPtr& context,
                   const Program::Ptr& intersectionProgram,
                   const Program::Ptr& boundingboxProgram,
                   size_t primitiveCount) :
    geometry_((*context)->createGeometry()),
    intersectionProgram_(intersectionProgram),
    boundingboxProgram_(boundingboxProgram)
{
    geometry_->setIntersectionProgram(*intersectionProgram_);
    geometry_->setBoundingBoxProgram(*boundingboxProgram_);
    geometry_->setPrimitiveCount(primitiveCount);
}

Program::Ptr Geometry::intersection_program()
{
    return intersectionProgram_;
}

Program::Ptr Geometry::boundingbox_program()
{
    return boundingboxProgram_;
}

Program::ConstPtr Geometry::intersection_program() const
{
    return intersectionProgram_;
}

Program::ConstPtr Geometry::boundingbox_program() const
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
