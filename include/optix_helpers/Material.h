#ifndef _DEF_OPTIX_HELPERS_MATERIAL_H_
#define _DEF_OPTIX_HELPERS_MATERIAL_H_

#include <iostream>
#include <unordered_map>
#include <vector>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/Program.h>
#include <optix_helpers/RayType.h>

namespace optix_helpers {

class Material
{
    //This is a class to help with material manipulations in optix
    public:

    using Ptr      = Handle<Material>;
    using ConstPtr = Handle<const Material>;

    using RayTypeCache = std::unordered_map<RayType::Index, RayType>;
    using ProgramCache = std::unordered_map<RayType::Index, Program::Ptr>;

    protected:
    
    optix::Material material_;
    RayTypeCache    rayTypes_;
    ProgramCache    closestHitPrograms_;
    ProgramCache    anyHitPrograms_;

    public:

    static Ptr New(const Context::ConstPtr& context);
    Material(const Context::ConstPtr& context);

    virtual Program::Ptr add_closest_hit_program(const RayType& rayType,
                                                 const Program::Ptr& program);
    virtual Program::Ptr add_any_hit_program(const RayType& rayType,
                                             const Program::Ptr& program);

    Program::ConstPtr get_closest_hit_program(const RayType& rayType) const;
    Program::ConstPtr get_any_hit_program(const RayType& rayType)     const;
    Program::Ptr get_closest_hit_program(const RayType& rayType);
    Program::Ptr get_any_hit_program(const RayType& rayType);
    optix::Material material() const;
    optix::Material operator->() const;
    operator optix::Material() const;
};
using Materials = std::vector<Material::ConstPtr>;

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_MATERIAL_H_

