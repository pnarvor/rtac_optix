#ifndef _DEF_RTAC_OPTIX_WRAPPER_H_
#define _DEF_RTAC_OPTIX_WRAPPER_H_

#include <iostream>

#include <rtac_base/types/BuildTarget.h>

#include <rtac_optix/utils.h>
#include <rtac_optix/Handle.h>

namespace rtac { namespace optix {

template <typename T>
using OptixWrapperHandle = rtac::types::BuildTargetHandle<T,Handle>;

template <typename OptixT>
class OptixWrapper : public rtac::types::BuildTarget
{
    // This type aims at being a generic helper to be used instead of native
    // OptiX types.
    // To use OptiX, a lot of different objects needs to be instanciated,
    // configured and built before being used for configuration of other OptiX
    // objects or directly inside OptiX API calls. It can be rather easy to
    // loose track of the order in which all these operations must be done, or
    // forgetting to perform an object build operation.
    // This object adresses theses issues using the build / dependency system
    // defined in the rtac::types::BuildTarget object. It allows to build a
    // tree of interdependent object which will trigger if needed the build
    // operation of its dependencies. It is also made implicitly castable to
    // the corresponding native OptiX type with auto build if needed for
    // seemsless use in OptiX API calls.
    public:

    using OptixType = OptixT;

    protected:
    
    // The optixObject_ is made mutable because of the design philosophy of
    // rtac::types::BuildTarget. In short, only the build parameters are
    // considered part of the observable state of the object. The build ouput
    // is merely considered a cache. This allows the build function to be
    // const.  The motivation is to allow dependent objects to hold a const
    // reference to this object but still being able to build it if necessary. 
    mutable OptixType optixObject_;

    OptixWrapper();

    public:

    operator OptixT&();
    operator const OptixT&() const;
};

template <typename OptixT>
OptixWrapper<OptixT>::OptixWrapper() : 
    BuildTarget(),
    optixObject_(types::zero<OptixType>())
{}

template <typename OptixT>
OptixWrapper<OptixT>::operator OptixT&()
{
    this->build();
    return optixObject_;
}

template <typename OptixT>
OptixWrapper<OptixT>::operator const OptixT&() const
{
    this->build();
    return optixObject_;
}

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_WRAPPER_H_
