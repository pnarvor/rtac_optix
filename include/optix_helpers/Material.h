#ifndef _DEF_OPTIX_HELPERS_MATERIAL_H_
#define _DEF_OPTIX_HELPERS_MATERIAL_H_

#include <iostream>
#include <unordered_map>

#include <optixu/optixpp.h>

#include <optix_helpers/Source.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>

namespace optix_helpers {

class Material
{
    //This is a class to help with material manipulations in optix
    public:

    using RayTypeCache = std::unordered_map<RayType::Index, RayType>;
    using SourceCache  = std::unordered_map<RayType::Index, Source>;
    using ProgramCache = std::unordered_map<RayType::Index, Program>;

    protected:
    
    Context         context_;
    optix::Material material_;
    RayTypeCache    rayTypes_;
    SourceCache     closestHitSources_;
    ProgramCache    closestHitPrograms_;
    SourceCache     anyHitSources_;
    ProgramCache    anyHitPrograms_;
    
    public:

    Material(const Context& context);

    Program add_closest_hit_program(const RayType& rayType, const Source& source,
                                    const Sources& additionalHeaders = Sources());
    Program add_any_hit_program(const RayType& rayType, const Source& source,
                                const Sources& additionalHeaders = Sources());

    optix::Material material() const;
    Source  get_closest_hit_source(const RayType& rayType)  const;
    Program get_closest_hit_program(const RayType& rayType) const;
    Source  get_any_hit_source(const RayType& rayType)      const;
    Program get_any_hit_program(const RayType& rayType)     const;
};

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_MATERIAL_H_
