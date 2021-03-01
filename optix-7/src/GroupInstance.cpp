#include <rtac_optix/GroupInstance.h>

namespace rtac { namespace optix {

GroupInstance::GroupInstance(const Context::ConstPtr& context) :
    Instance(InstanceAccelStruct::Create(context))
{}

GroupInstance::Ptr GroupInstance::Create(const Context::ConstPtr& context)
{
    return Ptr(new GroupInstance(context));
}

void GroupInstance::add_instance(const Instance::Ptr& instance)
{
    std::dynamic_pointer_cast<InstanceAccelStruct>(child_)->add_instance(instance);
}

}; //namespace optix
}; //namespace rtac
