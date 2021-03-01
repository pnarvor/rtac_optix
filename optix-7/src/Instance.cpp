#include <rtac_optix/Instance.h>

namespace rtac { namespace optix {

const float Instance::DefaultTransform[] = { 1.0f,0.0f,0.0f,0.0f,
                                             0.0f,1.0f,0.0f,0.0f,
                                             0.0f,0.0f,1.0f,0.0f };

OptixInstance Instance::default_instance(unsigned int instanceId)
{
    auto instance = zero<OptixInstance>();
    instance.visibilityMask = 255;
    instance.flags          = Instance::DefaultFlags;
    instance.instanceId     = instanceId;
    std::memcpy(&instance.transform,
                &Instance::DefaultTransform,
                12*sizeof(float));
    return instance;
}

Instance::Instance(const AccelerationStruct::Ptr& child,
                   unsigned int instanceId) :
    instance_(default_instance(instanceId)),
    child_(child)
{
    if(!child) {
        throw std::runtime_error("Instance::Instance : the child of an instance "
            "should not be nullptr. It cannot be modified afterwards.");
    }
}

Instance::Ptr Instance::Create(const AccelerationStruct::Ptr& child,
                               unsigned int instanceId)
{
    return Ptr(new Instance(child, instanceId));
}

void Instance::set_sbt_offset(unsigned int offset)
{
    instance_.sbtOffset = offset;
}

void Instance::set_visibility_mask(unsigned int mask)
{
    instance_.visibilityMask = mask;
}

void Instance::set_transform(const std::array<float,12>& transform)
{
    std::memcpy(&instance_.transform, transform.data(), 12*sizeof(float));
}

void Instance::set_flags(unsigned int flags)
{
    instance_.flags = flags;
}

void Instance::add_flags(unsigned int flag)
{
    instance_.flags |= flag;
}

void Instance::unset_flags(unsigned int flag)
{
    // invert all bits in flag then bitwise and will set flags to 0;
    instance_.flags &= ~flag; 
}

Instance::operator OptixInstance()
{
    if(!child_)
        instance_.traversableHandle = 0;
    else
        instance_.traversableHandle = *child_;
    return instance_;
}

OptixBuildInputType Instance::build_type() const
{
    if(!child_) {
        throw std::runtime_error("Instance : child is nullptr");
    }
    return child_->build_input().type;
}

Instance::operator OptixTraversableHandle()
{
    if(!child_)
        return 0;
    return *child_;
}

unsigned int Instance::sbt_width() const
{
    if(!child_)
        return 0;
    return child_->sbt_width();
}

}; //namespace optix
}; //namespace rtac

std::ostream& operator<<(std::ostream& os, const OptixInstance& instance)
{
    const char* prefix = "\n- ";
    os << "OptixInstance :"
       << prefix << "instanceId        : " << instance.instanceId
       << prefix << "sbtOffset         : " << instance.sbtOffset
       << prefix << "visibilityMask    : " << instance.visibilityMask
       << prefix << "traversableHandle : " << instance.traversableHandle
       << prefix << "flags             : " << instance.flags
       << prefix << "transform         : ";
    for(int i = 0; i < 8; i += 4) {
        os << instance.transform[i]     << " ";
        os << instance.transform[i + 1] << " ";
        os << instance.transform[i + 2] << " ";
        os << instance.transform[i + 3];
        os << prefix << "                    ";
    }
    os << instance.transform[8]     << " ";
    os << instance.transform[8 + 1] << " ";
    os << instance.transform[8 + 2] << " ";
    os << instance.transform[8 + 3];
    return os;
}

