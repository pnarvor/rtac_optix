#ifndef _DEF_OPTIX_HELPERS_RAY_TYPE_H_
#define _DEF_OPTIX_HELPERS_RAY_TYPE_H_

#include <iostream>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>

namespace optix_helpers {

class RayTypeObj
{
    public:

    using Index = int8_t;

    protected:

    Index  rayTypeIndex_; //Index of ray type as defined in optix kernels.
    Source definition_;   //Optix header defining the ray payload type.

    public:
    
    RayTypeObj(Index rayTypeIndex, const Source& definition);

    Index  index()      const;
    Source definition() const;
};

class RayType : public Handle<RayTypeObj>
{
    public:

    using Index = RayTypeObj::Index;

    RayType();
    RayType(Index rayTypeIndex, const Source& definition);
};

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::RayType& rayType);

#endif //_DEF_OPTIX_HELPERS_RAY_TYPE_H_
