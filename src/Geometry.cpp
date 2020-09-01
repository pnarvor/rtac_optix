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
    geometry_->setIntersectionProgram(intersectionProgram_);
    geometry_->setBoundingBoxProgram(boundingboxProgram_);
    this->set_primitive_count(primitiveCount);
}

void GeometryObj::set_intersection_program(const Program& program)
{
    intersectionProgram_ = program;
    geometry_->setIntersectionProgram(program);
}

void GeometryObj::set_boundingbox_program(const Program& program)
{
    boundingboxProgram_ = program;
    geometry_->setBoundingBoxProgram(program);
}

void GeometryObj::set_primitive_count(size_t primitiveCount)
{
    geometry_->setPrimitiveCount(primitiveCount);
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

size_t GeometryObj::primitive_count() const
{
    return geometry_->getPrimitiveCount();
}

Geometry::Geometry() :
    Handle<GeometryObj>()
{}

Geometry::Geometry(const optix::Geometry& geometry,
                   const Program& intersectionProgram,
                   const Program& boundingboxProgram,
                   size_t primitiveCount) :
    Handle<GeometryObj>(new GeometryObj(geometry, intersectionProgram,
                                        boundingboxProgram, primitiveCount))
{}

Geometry::operator optix::Geometry() const
{
    return (*this)->geometry();
}

}; //namespace optix_helpers
