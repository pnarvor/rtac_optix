#ifndef _DEF_OPTIX_HELPERS_MATERIAL_H_
#define _DEF_OPTIX_HELPERS_MATERIAL_H_

#include <iostream>
#include <unordered_map>
#include <vector>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Program.h>
#include <optix_helpers/RayType.h>

namespace optix_helpers {

class MaterialObj
{
    //This is a class to help with material manipulations in optix
    public:

    using RayTypeCache = std::unordered_map<RayType::Index, RayType>;
    using ProgramCache = std::unordered_map<RayType::Index, Program>;

    protected:
    
    optix::Material material_;
    RayTypeCache    rayTypes_;
    ProgramCache    closestHitPrograms_;
    ProgramCache    anyHitPrograms_;

    public:

    MaterialObj(const optix::Material& material);

    Program add_closest_hit_program(const RayType& rayType, const Program& program);
    Program add_any_hit_program(const RayType& rayType, const Program& program);

    optix::Material material() const;
    Program get_closest_hit_program(const RayType& rayType) const;
    Program get_any_hit_program(const RayType& rayType)     const;
};

class Material : public Handle<MaterialObj>
{
    public:

    Material();
    Material(const optix::Material& material);

    operator optix::Material() const;
};

using Materials = std::vector<Material>;


}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_MATERIAL_H_
