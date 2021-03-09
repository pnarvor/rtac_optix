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

Instance::Instance(const AccelerationStruct::ConstPtr& child,
                   unsigned int instanceId) :
    OptixWrapper<OptixInstance>(),
    child_(child)
{
    optixObject_ = default_instance(instanceId);
    if(!child) {
        throw std::runtime_error("Instance::Instance : the child of an instance "
            "should not be nullptr. It cannot be modified afterwards.");
    }
    this->add_dependency(child);
}

Instance::Ptr Instance::Create(const AccelerationStruct::ConstPtr& child,
                               unsigned int instanceId)
{
    return Ptr(new Instance(child, instanceId));
}

void Instance::do_build() const
{
    if(this->child_)
        optixObject_.traversableHandle = *child_;
    else
        optixObject_.traversableHandle = 0;
}

void Instance::set_sbt_offset(unsigned int offset)
{
    optixObject_.sbtOffset = offset;
    this->bump_version(false);
}

void Instance::set_visibility_mask(unsigned int mask)
{
    optixObject_.visibilityMask = mask;
    this->bump_version(false);
}

void Instance::set_transform(const std::array<float,12>& transform)
{
    std::memcpy(&optixObject_.transform, transform.data(), 12*sizeof(float));
    this->bump_version(false);
}

void Instance::set_flags(unsigned int flags)
{
    optixObject_.flags = flags;
    this->bump_version(false);
}

void Instance::add_flags(unsigned int flag)
{
    optixObject_.flags |= flag;
    this->bump_version(false);
}

void Instance::unset_flags(unsigned int flag)
{
    // invert all bits in flag then bitwise and will set flags to 0;
    optixObject_.flags &= ~flag; 
    this->bump_version(false);
}

unsigned int Instance::sbt_width() const
{
    if(!child_)
        return 0;
    return child_->sbt_width();
}

Instance::operator const OptixTraversableHandle&() const
{
    if(!child_) {
        throw std::runtime_error("This instance has no child !");
    }
    return *child_;
}

// below here are only helpers / overrides
void Instance::set_position(const std::array<float,3>& pos)
{
    optixObject_.transform[3]  = pos[0];
    optixObject_.transform[7]  = pos[1];
    optixObject_.transform[11] = pos[2];
    this->bump_version(false);
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

