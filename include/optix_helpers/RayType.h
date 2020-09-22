#ifndef _DEF_OPTIX_HELPERS_RAY_TYPE_H_
#define _DEF_OPTIX_HELPERS_RAY_TYPE_H_

#include <iostream>
#include <memory>
#include <climits>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>

namespace optix_helpers {

class RayTypeObj
{
    public:

    using Index = unsigned int;
    static const Index uninitialized = UINT_MAX;

    protected:

    Index  rayTypeIndex_; //Index of ray type as defined in optix kernels.
    Source definition_;   //Optix header defining the ray payload type.

    public:

    RayTypeObj(Index rayTypeIndex, const Source& definition);

    Source definition() const;
    Index    index()    const;
    operator Index()    const;
};

class RayType : public Handle<RayTypeObj>
{
    public:
    
    using Index = RayTypeObj::Index;
    static const Index uninitialized = RayTypeObj::uninitialized;

    RayType();
    RayType(Index rayTypeIndex, const Source& definition);

    operator Index() const;
};

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::RayType& rayType);

#endif //_DEF_OPTIX_HELPERS_RAY_TYPE_H_
