#ifndef _DEF_RTAC_OPTIX_INSTANCE_H_
#define _DEF_RTAC_OPTIX_INSTANCE_H_

#include <iostream>
#include <cstring>
#include <array>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>
#include <rtac_optix/TraversableHandle.h>
#include <rtac_optix/AccelerationStruct.h>
#include <rtac_optix/Material.h>

namespace rtac { namespace optix {

class Instance : public TraversableHandle
{
    public:

    using Ptr      = Handle<Instance>;
    using ConstPtr = Handle<const Instance>;

    const static unsigned int DefaultFlags = OPTIX_INSTANCE_FLAG_NONE;
    const static float DefaultTransform[];

    static OptixInstance default_instance(unsigned int instanceId = 0);

    protected:
    
    OptixInstance           instance_;
    AccelerationStruct::Ptr child_;

    Instance(const AccelerationStruct::Ptr& handle,
             unsigned int instanceId = 0);

    public:

    static Ptr Create(const AccelerationStruct::Ptr& child,
                      unsigned int instanceId = 0); // what is instanceId for ?

    void set_sbt_offset(unsigned int offset);
    void set_visibility_mask(unsigned int mask);
    void set_transform(const std::array<float,12>& transform);

    void set_flags(unsigned int flags);
    void add_flags(unsigned int flag);
    void unset_flags(unsigned int flag);

    operator OptixInstance();
    
    OptixBuildInputType build_type() const;
    virtual operator OptixTraversableHandle();
    virtual unsigned int sbt_width() const;
};

}; //namespace optix
}; //namespace rtac

std::ostream& operator<<(std::ostream& os, const OptixInstance& instance);



#endif //_DEF_RTAC_OPTIX_INSTANCE_H_
