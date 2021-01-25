#include <optix_helpers/Material.h>

namespace optix_helpers {

Material::Ptr Material::New(const Context::ConstPtr& context)
{
    return Ptr(new Material(context));
}

Material::Material(const Context::ConstPtr& context) :
    material_((*context)->createMaterial())
{}

Program::Ptr Material::add_closest_hit_program(const RayType& rayType,
                                                    const Program::Ptr& program)
{
    material_->setClosestHitProgram(rayType.index(), *program);
    closestHitPrograms_.insert({rayType.index(), program});
    rayTypes_.insert({rayType.index(), rayType});

    return program;
}

Program::Ptr Material::add_any_hit_program(const RayType& rayType,
                                                const Program::Ptr& program)
{
    material_->setAnyHitProgram(rayType.index(), *program);
    anyHitPrograms_.insert({rayType.index(), program});
    rayTypes_.insert({rayType.index(), rayType});

    return program;
}

Program::ConstPtr Material::get_closest_hit_program(const RayType& rayType) const
{
    return closestHitPrograms_.at(rayType.index());
}

Program::ConstPtr Material::get_any_hit_program(const RayType& rayType) const
{
    return anyHitPrograms_.at(rayType.index());
}

Program::Ptr Material::get_closest_hit_program(const RayType& rayType)
{
    return closestHitPrograms_.at(rayType.index());
}

Program::Ptr Material::get_any_hit_program(const RayType& rayType)
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

