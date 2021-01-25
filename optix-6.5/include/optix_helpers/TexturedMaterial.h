#ifndef _DEF_OPTIX_HELPERS_TEXTURED_MATERIAL_H_
#define _DEF_OPTIX_HELPERS_TEXTURED_MATERIAL_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/TextureSampler.h>
#include <optix_helpers/Material.h>

namespace optix_helpers {

class TexturedMaterial : public Material
{
    public:
    
    using Ptr      = Handle<TexturedMaterial>;
    using ConstPtr = Handle<const TexturedMaterial>;

    protected:

    TextureSampler::ConstPtr texture_;

    public:

    static Ptr New(const Context::ConstPtr& context,
                   const TextureSampler::ConstPtr& texture);
    TexturedMaterial(const Context::ConstPtr& context,
                     const TextureSampler::ConstPtr& texture);

    virtual Program::Ptr add_closest_hit_program(const RayType& rayType,
                                                 const Program::Ptr& program);
    virtual Program::Ptr add_any_hit_program(const RayType& rayType,       
                                             const Program::Ptr& program);

    TextureSampler::ConstPtr texture() const;
};

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_TEXTURED_MATERIAL_H_
