#include <optix_helpers/Material.h>

namespace optix_helpers {

Material::Material(const optix::Material& material) :
    material_(material)
{}

Program Material::add_closest_hit_program(const RayType& rayType, const Program& program)
{
    material_->setClosestHitProgram(rayType.index(), program);
    closestHitPrograms_.insert({rayType.index(), program});
    rayTypes_.insert({rayType.index(), rayType});

    return program;
}

Program Material::add_any_hit_program(const RayType& rayType, const Program& program)
{
    material_->setAnyHitProgram(rayType.index(), program);
    anyHitPrograms_.insert({rayType.index(), program});
    rayTypes_.insert({rayType.index(), rayType});

    return program;
}

Program Material::get_closest_hit_program(const RayType& rayType) const
{
    return closestHitPrograms_.at(rayType.index());
}

Program Material::get_any_hit_program(const RayType& rayType) const
{
    return anyHitPrograms_.at(rayType.index());
}

optix::Material Material::material() const
{
    return material_;
}

optix::Material Material::operator->() const
{
    return material_;
}

Material::operator optix::Material() const
{
    return material_;
}

}; //namespace optix_helpers

