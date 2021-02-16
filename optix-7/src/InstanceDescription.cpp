#include <rtac_optix/InstanceDescription.h>

namespace rtac { namespace optix {

const float InstanceDescription::DefaultTransform[] = { 1.0f,0.0f,0.0f,0.0f,
                                                        0.0f,1.0f,0.0f,0.0f,
                                                        0.0f,0.0f,1.0f,0.0f };

OptixInstance InstanceDescription::default_instance()
{
    auto instance = zero<OptixInstance>();
    instance.visibilityMask = 255;
    instance.flags          = InstanceDescription::DefaultFlags;
    std::memcpy(&instance.transform,
                &InstanceDescription::DefaultTransform,
                12*sizeof(float));
    return instance;
}

InstanceDescription::InstanceDescription(unsigned int instanceId,
                                         const OptixTraversableHandle& handle) :
    instance_(default_instance())
{
    instance_.instanceId        = instanceId;
    instance_.traversableHandle = handle;
}

InstanceDescription::Ptr InstanceDescription::Create(unsigned int instanceId,
                                                     OptixTraversableHandle handle)
{
    return Ptr(new InstanceDescription(instanceId, handle));
}

void InstanceDescription::set_traversable_handle(const OptixTraversableHandle& handle)
{
    instance_.traversableHandle = handle;
}

void InstanceDescription::set_sbt_offset(unsigned int offset)
{
    instance_.sbtOffset = offset;
}

void InstanceDescription::set_visibility_mask(unsigned int mask)
{
    instance_.visibilityMask = mask;
}

void InstanceDescription::set_transform(const std::array<float,12>& transform)
{
    std::memcpy(&instance_.transform, transform.data(), 12*sizeof(float));
}

void InstanceDescription::set_flags(unsigned int flags)
{
    instance_.flags = flags;
}

void InstanceDescription::add_flags(unsigned int flag)
{
    instance_.flags |= flag;
}

void InstanceDescription::unset_flags(unsigned int flag)
{
    // invert all bits in flag then bitwise and will set flags to 0;
    instance_.flags &= ~flag; 
}

InstanceDescription::operator OptixInstance()
{
    return instance_;
}

InstanceDescription::operator OptixInstance() const
{
    return instance_;
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

