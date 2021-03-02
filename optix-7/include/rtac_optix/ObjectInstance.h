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

    using Ptr       = Handle<ObjectInstance>;
    using ConstPtr  = Handle<const ObjectInstance>;
    using Materials = std::vector<std::pair<unsigned int, MaterialBase::Ptr>>;
    
    protected:

    Materials materials_;

    ObjectInstance(const GeometryAccelStruct::Ptr& geometry,
                   unsigned int instanceId = 0); // What is it for ?

    public:

    static Ptr Create(const GeometryAccelStruct::Ptr& geometry,
                      unsigned int instanceId = 0); // What is it for ?
    
    // num_materials if given by the GeometryAccelStruct (several Instances can
    // share the same GeometryAccelStruct, so it is the responsibility of
    // GeometryAccelStruct to set the number of materials).
    unsigned int material_count() const;
    //MaterialBase::ConstPtr material(unsigned int index) const;
    //MaterialBase::Ptr      material(unsigned int index);
    //void set_material(const MaterialBase::Ptr& material,
    //                  unsigned int index = 0);
    //void unset_material(unsigned int index);
    void add_material(const MaterialBase::Ptr& material,
                      unsigned int index = 0);
    Materials& materials(); // bad. Do this better
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_OBJECT_INSTANCE_H_
