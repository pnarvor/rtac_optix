#include <optix_helpers/Material.h>

namespace optix_helpers {

MaterialObj::MaterialObj(const Context& context) :
    material_((*context)->createMaterial())
{}

Program MaterialObj::add_closest_hit_program(const RayType& rayType, const Program& program)
{
    material_->setClosestHitProgram(rayType->index(), *program);
    closestHitPrograms_.insert({rayType->index(), program});
    rayTypes_.insert({rayType->index(), rayType});

    return program;
}

Program MaterialObj::add_any_hit_program(const RayType& rayType, const Program& program)
{
    material_->setAnyHitProgram(rayType->index(), *program);
    anyHitPrograms_.insert({rayType->index(), program});
    rayTypes_.insert({rayType->index(), rayType});

    return program;
}

Program MaterialObj::get_closest_hit_program(const RayType& rayType) const
{
    return closestHitPrograms_.at(rayType->index());
}

Program MaterialObj::get_any_hit_program(const RayType& rayType) const
{
    return anyHitPrograms_.at(rayType->index());
}

optix::Material MaterialObj::material() const
{
    return material_;
}

optix::Material MaterialObj::operator->() const
{
    return material_;
}

MaterialObj::operator optix::Material() const
{
    return material_;
}

}; //namespace optix_helpers

