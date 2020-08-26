#include <optix_helpers/Material.h>

namespace optix_helpers {

Material::Material(const Context& context) :
    context_(context),
    material_(context_.context()->createMaterial())
{
}

Program Material::add_closest_hit_program(const RayType& rayType, const Source& source,
                                          const Sources& additionalHeaders)
{
    Program program(context_.create_program(source, additionalHeaders));
    material_->setClosestHitProgram(rayType.index(), program.program());
    
    closestHitSources_[rayType.index()]  = source;
    closestHitPrograms_[rayType.index()] = program;
    rayTypes_[rayType.index()]           = rayType;

    return program;
}

Program Material::add_any_hit_program(const RayType& rayType, const Source& source,
                                          const Sources& additionalHeaders)
{
    Program program(context_.create_program(source, additionalHeaders));
    material_->setAnyHitProgram(rayType.index(), program.program());
    
    anyHitSources_[rayType.index()]  = source;
    anyHitPrograms_[rayType.index()] = program;
    rayTypes_[rayType.index()]       = rayType;

    return program;
}

optix::Material Material::material() const
{
    return material_;
}

Source Material::get_closest_hit_source(const RayType& rayType) const
{
    return closestHitSources_.at(rayType.index());
}

Program Material::get_closest_hit_program(const RayType& rayType) const
{
    return closestHitPrograms_.at(rayType.index());
}

Source Material::get_any_hit_source(const RayType& rayType) const
{
    return anyHitSources_.at(rayType.index());
}

Program Material::get_any_hit_program(const RayType& rayType) const
{
    return anyHitPrograms_.at(rayType.index());
}




}; //namespace optix_helpers

