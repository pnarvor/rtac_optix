#include <optix_helpers/TexturedMaterial.h>

namespace optix_helpers {

TexturedMaterial::Ptr TexturedMaterial::New(const Context::ConstPtr& context,
                                            const TextureSampler::ConstPtr& texture)
{
    return Ptr(new TexturedMaterial(context, texture));
}

TexturedMaterial::TexturedMaterial(const Context::ConstPtr& context,
                                   const TextureSampler::ConstPtr& texture) :
    Material(context),
    texture_(texture)
{
}

Program::Ptr TexturedMaterial::add_closest_hit_program(const RayType& rayType,
                                                       const Program::Ptr& program)
{
    this->Material::add_closest_hit_program(rayType, program)->set_object(texture_);
    return program;
}

Program::Ptr TexturedMaterial::add_any_hit_program(const RayType& rayType,
                                                   const Program::Ptr& program)
{
    this->Material::add_any_hit_program(rayType, program)->set_object(texture_);
    return program;
}

TextureSampler::ConstPtr TexturedMaterial::texture() const
{
    return texture_;
}

};//namespace optix_helpers
