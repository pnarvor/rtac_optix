#include <rtac_optix/InstanceAccelStruct.h>

namespace rtac { namespace optix {

InstanceAccelStruct::BuildInput InstanceAccelStruct::default_build_input()
{
    auto res = zero<BuildInput>();
    res.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    return res;
}

InstanceAccelStruct::BuildOptions InstanceAccelStruct::default_build_options()
{
    return AccelerationStruct::default_build_options();
}

InstanceAccelStruct::InstanceAccelStruct(const Context::ConstPtr& context) :
    AccelerationStruct(context, default_build_input(), default_build_options())
{}

InstanceAccelStruct::Ptr InstanceAccelStruct::Create(const Context::ConstPtr& context)
{
    return Ptr(new InstanceAccelStruct(context));
}

void InstanceAccelStruct::build()
{
    // The buildInput_ needs to be updated before the build with created
    // instances.
    std::vector<OptixInstance> instances(instances_.size());
    for(int i = 0; i < instances.size(); i++) {
        instances[i] = *instances_[i];
    }
    tmpInstanceData_ = instances; // This uploads instances on the device.
    
    this->buildInput_.instanceArray.instances =
        reinterpret_cast<CUdeviceptr>(tmpInstanceData_.data());
    this->buildInput_.instanceArray.numInstances = tmpInstanceData_.size();

    this->AccelerationStruct::build();

    cudaDeviceSynchronize();
}

void InstanceAccelStruct::add_instance(const Instance::Ptr& instance)
{
    instances_.push_back(instance);
}

InstanceAccelStruct::Instances& InstanceAccelStruct::instances()
{
    return instances_;
}

const InstanceAccelStruct::Instances& InstanceAccelStruct::instances() const
{
    return instances_;
}

unsigned int InstanceAccelStruct::sbt_width() const
{
    unsigned int res = 0;
    for(auto& instance : instances_) {
        res += instance->sbt_width();
    }
    return res;
}

}; //namespace optix
}; //namespace rtac
