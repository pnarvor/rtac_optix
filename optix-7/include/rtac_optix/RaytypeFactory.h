#ifndef _DEF_RTAC_OPTIX_RAYTYPE_FACTORY_H_
#define _DEF_RTAC_OPTIX_RAYTYPE_FACTORY_H_

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <std::tuple>

namespace rtac { namespace optix {

template <class... RayPayloadTs>
class RaytypeFactory
{
    // This is a class intended to generate the Raytypes to be used both on
    // host-side on device-side functions.

    // Raytypes get unique ids which are used when tracing a ray in device-side
    // functions. These ids are also used on host  to generate the SBT (Shader
    // Binding Table). Similarly, the total number on Raytypes is used on both
    // side. This class together with Raytype and Material related type allow
    // to free the user from these considerations.
    
    public:

    using PayloadTypes = std::tuple<RayPayloadT...>;

};

}; //namespace optix
}; //namespace rtac


#endif //_DEF_RTAC_OPTIX_RAYTYPE_FACTORY_H_
