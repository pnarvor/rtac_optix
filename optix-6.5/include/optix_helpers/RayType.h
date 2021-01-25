#ifndef _DEF_OPTIX_HELPERS_RAY_TYPE_H_
#define _DEF_OPTIX_HELPERS_RAY_TYPE_H_

#include <iostream>
#include <memory>
#include <climits>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>

namespace optix_helpers {

// forward declaration for allowing only Context to build RayTypes.
class Context;

class RayType
{
    public:

    using Index = unsigned int;
    static constexpr const Index uninitialized = UINT_MAX;
    friend class Context;

    protected:

    Index            rayTypeIndex_; //Index of ray type as defined in optix kernels.
    Source::ConstPtr definition_;   //Optix header defining the ray payload type.
    
    private:

    RayType(Index rayTypeIndex, const Source::ConstPtr& definition);

    public:

    Index            index()    const;
    operator         Index()    const;
    Source::ConstPtr definition() const;
};

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::RayType& rayType);

#endif //_DEF_OPTIX_HELPERS_RAY_TYPE_H_
