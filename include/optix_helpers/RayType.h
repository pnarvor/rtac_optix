#ifndef _DEF_OPTIX_HELPERS_RAY_TYPE_H_
#define _DEF_OPTIX_HELPERS_RAY_TYPE_H_

#include <iostream>
#include <memory>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>

namespace optix_helpers {

// Forward declaration for Context as friend of RayType
class ContextObj;

class RayTypeObj
{
    public:

    using Index = unsigned int;

    protected:

    Index  rayTypeIndex_; //Index of ray type as defined in optix kernels.
    Source definition_;   //Optix header defining the ray payload type.

    RayTypeObj(Index rayTypeIndex, const Source& definition);

    public:

    Index  index()      const;
    Source definition() const;

    friend class ContextObj; // Make Context only creator of RayTypes
};
using RayType      = std::shared_ptr<RayTypeObj>;
using RayTypeIndex = RayTypeObj::Index;

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::RayType& rayType);

#endif //_DEF_OPTIX_HELPERS_RAY_TYPE_H_
