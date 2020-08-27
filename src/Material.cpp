#include <optix_helpers/Material.h>

namespace optix_helpers {

MaterialObj::MaterialObj(const Context& context) :
    context_(context),
    material_(context_->context()->createMaterial())
{
}

Program MaterialObj::compile(const RayType& rayType, const Source& source,
                             const Sources& additionalHeaders) const
{
    Sources headers(additionalHeaders);
    headers.push_back(rayType->definition());

    return context_->create_program(source, headers);
}

Program MaterialObj::add_closest_hit_program(const RayType& rayType, const Source& source,
                                             const Sources& additionalHeaders)
{
    Program program(this->compile(rayType, source, additionalHeaders));
    material_->setClosestHitProgram(rayType->index(), program);
    
    closestHitSources_[rayType->index()]  = source;
    closestHitPrograms_[rayType->index()] = program;
    rayTypes_[rayType->index()]           = rayType;

    return program;
}

Program MaterialObj::add_any_hit_program(const RayType& rayType, const Source& source,
                                         const Sources& additionalHeaders)
{
    Program program(this->compile(rayType, source, additionalHeaders));
    material_->setAnyHitProgram(rayType->index(), program);
    
    anyHitSources_[rayType->index()]  = source;
    anyHitPrograms_[rayType->index()] = program;
    rayTypes_[rayType->index()]       = rayType;

    return program;
}

optix::Material MaterialObj::material() const
{
    return material_;
}

Source MaterialObj::get_closest_hit_source(const RayType& rayType) const
{
    return closestHitSources_.at(rayType->index());
}

Program MaterialObj::get_closest_hit_program(const RayType& rayType) const
{
    return closestHitPrograms_.at(rayType->index());
}

Source MaterialObj::get_any_hit_source(const RayType& rayType) const
{
    return anyHitSources_.at(rayType->index());
}

Program MaterialObj::get_any_hit_program(const RayType& rayType) const
{
    return anyHitPrograms_.at(rayType->index());
}

Material::Material() :
    Handle<MaterialObj>()
{}

Material::Material(const Context& context) :
    Handle<MaterialObj>(new MaterialObj(context))
{}

}; //namespace optix_helpers

