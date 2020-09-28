#include <optix_helpers/TexturedMaterial.h>

namespace optix_helpers {

TexturedMaterialObj::TexturedMaterialObj(const Context& context, const TextureSampler& texture) :
    MaterialObj(context),
    texture_(texture)
{
}

Program TexturedMaterialObj::add_closest_hit_program(const RayType& rayType,
                                                     const Program& program)
{
    this->MaterialObj::add_closest_hit_program(rayType, program)->set_object(texture_);
    return program;
}

Program TexturedMaterialObj::add_any_hit_program(const RayType& rayType,
                                                 const Program& program)
{
    this->MaterialObj::add_any_hit_program(rayType, program)->set_object(texture_);
    return program;
}

TextureSampler TexturedMaterialObj::texture()
{
    return texture_;
}

const TextureSampler TexturedMaterialObj::texture() const
{
    return texture_;
}

//TexturedMaterial::TexturedMaterial() :
//    Handle<TexturedMaterialObj>()
//{}
//
//TexturedMaterial::TexturedMaterial(const Context& context,
//                                   const TextureSampler& texture) :
//    Handle<TexturedMaterialObj>(context, texture)
//{}
//
//TexturedMaterial::operator Material()
//{
//    return Material(std::static_pointer_cast<MaterialObj>(this->obj_));
//}
//
//TexturedMaterial::operator Material() const
//{
//    return Material(std::static_pointer_cast<MaterialObj>(this->obj_));
//}

};//namespace optix_helpers
