#include <rtac_optix/GroupInstance.h>

namespace rtac { namespace optix {

GroupInstance::GroupInstance(const Context::ConstPtr& context) :
    Instance(InstanceAccelStruct::Create(context))
{}

GroupInstance::Ptr GroupInstance::Create(const Context::ConstPtr& context)
{
    return Ptr(new GroupInstance(context));
}

InstanceAccelStruct::Ptr GroupInstance::instanceAS()
{
    return std::dynamic_pointer_cast<InstanceAccelStruct>(child_);
}

InstanceAccelStruct::ConstPtr GroupInstance::instanceAS() const
{
    return std::dynamic_pointer_cast<const InstanceAccelStruct>(child_);
}

void GroupInstance::add_instance(const Instance::Ptr& instance)
{
    this->instanceAS()->add_instance(instance);
}

GroupInstance::Instances& GroupInstance::instances()
{
    return this->instanceAS()->instances();
}

const GroupInstance::Instances& GroupInstance::instances() const
{
    return this->instanceAS()->instances();
}

}; //namespace optix
}; //namespace rtac
