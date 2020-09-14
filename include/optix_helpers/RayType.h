#ifndef _DEF_OPTIX_HELPERS_RAY_TYPE_H_
#define _DEF_OPTIX_HELPERS_RAY_TYPE_H_

#include <iostream>
#include <memory>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>

namespace optix_helpers {

// Forward declaration for Context as friend of RayType
class ContextObj;
class RayType;

class RayTypeObj
{
    public:

    using Index = unsigned int;

    protected:

    Index  rayTypeIndex_; //Index of ray type as defined in optix kernels.
    Source definition_;   //Optix header defining the ray payload type.

    RayTypeObj(Index rayTypeIndex, const Source& definition);

    public:

    Source definition() const;
    Index    index()    const;
    operator Index()    const;

    friend class ContextObj; // Make Context only creator of RayTypes
    friend class RayType;
};

class RayType : public Handle<RayTypeObj>
{
    public:

    using Index = RayTypeObj::Index;

    RayType();

    operator Index() const;

    protected:

    RayType(Index rayTypeIndex, const Source& definition);

    friend class ContextObj; // Make Context only creator of RayTypes
};

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::RayType& rayType);

#endif //_DEF_OPTIX_HELPERS_RAY_TYPE_H_
