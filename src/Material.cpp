#include <optix_helpers/Material.h>

namespace optix_helpers {

MaterialObj::MaterialObj(const optix::Material& material) :
    material_(material)
{
}

Program MaterialObj::add_closest_hit_program(const RayType& rayType, const Program& program)
{
    material_->setClosestHitProgram(rayType->index(), program);
    closestHitPrograms_[rayType->index()] = program;
    rayTypes_[rayType->index()]           = rayType;

    return program;
}

Program MaterialObj::add_any_hit_program(const RayType& rayType, const Program& program)
{
    material_->setAnyHitProgram(rayType->index(), program);
    anyHitPrograms_[rayType->index()] = program;
    rayTypes_[rayType->index()]       = rayType;

    return program;
}

optix::Material MaterialObj::material() const
{
    return material_;
}

Program MaterialObj::get_closest_hit_program(const RayType& rayType) const
{
    return closestHitPrograms_.at(rayType->index());
}

Program MaterialObj::get_any_hit_program(const RayType& rayType) const
{
    return anyHitPrograms_.at(rayType->index());
}

Material::Material() :
    Handle<MaterialObj>()
{}

Material::Material(const optix::Material& material) :
    Handle<MaterialObj>(new MaterialObj(material))
{}

}; //namespace optix_helpers

