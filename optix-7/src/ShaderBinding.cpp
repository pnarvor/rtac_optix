#include <rtac_optix/ShaderBinding.h>

namespace rtac { namespace optix {

ShaderBindingBase::ShaderBindingBase(const ProgramGroup::Ptr& program) :
    program_(program),
    needsUpdate_(true)
{}

bool ShaderBindingBase::needs_update() const
{
    return needsUpdate_;
}

ProgramGroup::Ptr ShaderBindingBase::program()
{
    return program_;
}

ProgramGroup::ConstPtr ShaderBindingBase::program() const
{
    return program_;
}

}; //namespace optix
}; //namespace rtac
