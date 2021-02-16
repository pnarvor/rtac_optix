#ifndef _DEF_RTAC_OPTIX_INSTANCE_DESCRIPTION_H_
#define _DEF_RTAC_OPTIX_INSTANCE_DESCRIPTION_H_

#include <iostream>
#include <cstring>
#include <array>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>

namespace rtac { namespace optix {

// Forward declaration of InstanceAccelStruct to make it the only type allowed
// to instanciate a new InstanceDescription.
class InstanceAccelStruct;

class InstanceDescription
{
    public:

    friend class InstanceAccelStruct;

    using Ptr      = Handle<InstanceDescription>;
    using ConstPtr = Handle<const InstanceDescription>;

    const static unsigned int DefaultFlags = OPTIX_INSTANCE_FLAG_NONE;
    const static float DefaultTransform[];

    static OptixInstance default_instance();

    protected:

    OptixInstance instance_;

    InstanceDescription(unsigned int instanceId,
                        const OptixTraversableHandle& handle = 0);

    static Ptr Create(unsigned int instanceId,
                      OptixTraversableHandle handle = 0);

    public:

    void set_traversable_handle(const OptixTraversableHandle& handle);
    void set_sbt_offset(unsigned int offset);
    void set_visibility_mask(unsigned int mask);
    void set_transform(const std::array<float,12>& transform);
    void set_flags(unsigned int flags);
    void add_flags(unsigned int flag);
    void unset_flags(unsigned int flag);

    operator OptixInstance();
    operator OptixInstance() const;
};

}; //namespace optix
}; //namespace rtac

std::ostream& operator<<(std::ostream& os, const OptixInstance& instance);



#endif //_DEF_RTAC_OPTIX_INSTANCE_DESCRIPTION_H_
