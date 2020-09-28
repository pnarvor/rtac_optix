#ifndef _DEF_OPTIX_HELPERS_TEXTURED_MATERIAL_H_
#define _DEF_OPTIX_HELPERS_TEXTURED_MATERIAL_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/TextureSampler.h>
#include <optix_helpers/Material.h>

namespace optix_helpers {

class TexturedMaterialObj : public MaterialObj
{
    protected:

    TextureSampler texture_;

    public:

    TexturedMaterialObj(const Context& context,
                        const TextureSampler& texture);

    virtual Program add_closest_hit_program(const RayType& rayType, const Program& program);
    virtual Program add_any_hit_program(const RayType& rayType, const Program& program);

    TextureSampler texture();
    const TextureSampler texture() const;
};

using TexturedMaterial = Handle<TexturedMaterialObj>;
//class TexturedMaterial : public Handle<TexturedMaterialObj>
//{
//    public:
//
//    TexturedMaterial();
//    TexturedMaterial(const Context& context,
//                     const TextureSampler& texture);
//
//    operator Material();
//    operator Material() const;
//};

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_TEXTURED_MATERIAL_H_
