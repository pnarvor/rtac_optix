#ifndef _DEF_RTAC_OPTIX_OBJECT_INSTANCE_H_
#define _DEF_RTAC_OPTIX_OBJECT_INSTANCE_H_

#include <iostream>
#include <vector>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>
#include <rtac_optix/GeometryAccelStruct.h>
#include <rtac_optix/Instance.h>
#include <rtac_optix/Material.h>

namespace rtac { namespace optix {

class ObjectInstance : public Instance
{
    // This class represent a single object in the world (geometry +
    // material).

    public:

    using Ptr       = OptixWrapperHandle<ObjectInstance>;
    using ConstPtr  = OptixWrapperHandle<const ObjectInstance>;
    using Materials = std::vector<std::pair<unsigned int, MaterialBase::ConstPtr>>;
    
    protected:

    Materials materials_;

    ObjectInstance(const GeometryAccelStruct::ConstPtr& geometry,
                   unsigned int instanceId = 0);

    public:

    static Ptr Create(const GeometryAccelStruct::ConstPtr& geometry,
                      unsigned int instanceId = 0);
    
    unsigned int material_count() const;
    void add_material(const MaterialBase::ConstPtr& material,
                      unsigned int index = 0);
    const Materials& materials() const;
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_OBJECT_INSTANCE_H_
