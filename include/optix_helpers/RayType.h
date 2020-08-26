#ifndef _DEF_OPTIX_HELPERS_RAY_TYPE_H_
#define _DEF_OPTIX_HELPERS_RAY_TYPE_H_

#include <iostream>

#include <optix_helpers/Source.h>

namespace optix_helpers {

class RayType
{
    public:

    using Index = int8_t;

    protected:

    Index  rayTypeIndex_; //Index of ray type as defined in optix kernels.
    Source definition_; //Optix header defining the ray payload type.

    public:
    
    RayType(Index rayTypeIndex = -1, const Source& definition = Source());
    RayType(const RayType& other);

    Index  index()      const;
    Source definition() const;
    bool   is_defined() const;
};

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_RAY_TYPE_H_
