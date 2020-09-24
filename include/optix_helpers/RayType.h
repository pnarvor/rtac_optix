#ifndef _DEF_OPTIX_HELPERS_RAY_TYPE_H_
#define _DEF_OPTIX_HELPERS_RAY_TYPE_H_

#include <iostream>
#include <memory>
#include <climits>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>

namespace optix_helpers {

// forward declaration for allowing RayType to access RayTypeObj constructor
class RayType;

class RayTypeObj
{
    public:

    using Index = unsigned int;
    static const Index uninitialized = UINT_MAX;
    friend class RayType; // To allow RayType to instanciate a RayTypeObj

    protected:

    Index  rayTypeIndex_; //Index of ray type as defined in optix kernels.
    Source definition_;   //Optix header defining the ray payload type.
    
    private:

    RayTypeObj(Index rayTypeIndex, const Source& definition);

    public:

    Source definition() const;
    Index    index()    const;
    operator Index()    const;

};

// forward declaration for allowing only ContextObj to build RayTypes.
class ContextObj;
class RayType : public Handle<RayTypeObj>
{
    public:
    
    using Index = RayTypeObj::Index;
    static const Index uninitialized = RayTypeObj::uninitialized;
    friend class ContextObj;

    RayType();

    operator Index() const;
    
    private:

    RayType(Index rayTypeIndex, const Source& definition);
};

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::RayType& rayType);

#endif //_DEF_OPTIX_HELPERS_RAY_TYPE_H_
