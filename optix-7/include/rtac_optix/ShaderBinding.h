#ifndef _DEF_RTAC_OPTIX_SHADER_BINDING_H_
#define _DEF_RTAC_OPTIX_SHADER_BINDING_H_

#include <iostream>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>
#include <rtac_optix/OptixWrapper.h>
#include <rtac_optix/ProgramGroup.h>

namespace rtac { namespace optix {

class ShaderBindingBase
{
    public:

    using Ptr      = OptixWrapperHandle<ShaderBindingBase>;
    using ConstPtr = OptixWrapperHandle<const ShaderBindingBase>;

    protected:

    // putting these mutable for now. To be fixed.
    mutable ProgramGroup::Ptr program_;

    ShaderBindingBase(const ProgramGroup::Ptr& program);
    
    public:

    ProgramGroup::Ptr      program();
    ProgramGroup::ConstPtr program() const;

    virtual unsigned int record_size() const = 0;
    virtual void fill_sbt_record(void* dst) const = 0;
};

template <typename ParamsT>
class ShaderBinding : public OptixWrapper<SbtRecord<ParamsT>>,
                      public virtual ShaderBindingBase
{
    public:

    using ParamsType    = ParamsT;
    using SbtRecordType = SbtRecord<ParamsT>;

    using Ptr      = OptixWrapperHandle<ShaderBinding<ParamsType>>;
    using ConstPtr = OptixWrapperHandle<const ShaderBinding<ParamsType>>;
    
    protected:

    void do_build() const;

    // Putting a SbtRecordType as constructor parameter to be compatible with
    // void ParamT SbtRecord are constructible with a ParamT, so Create can
    // still be called directly with parameters.
    ShaderBinding(const ProgramGroup::Ptr& program, const SbtRecordType& params);

    public:
    
    static Ptr Create(const ProgramGroup::Ptr& program,
                      const SbtRecordType& params = zero<SbtRecordType>());
    
    SbtRecordType& record();
    const SbtRecordType& record() const;

    virtual unsigned int record_size() const;
    virtual void fill_sbt_record(void* dst) const;
};

template <typename ParamsT>
ShaderBinding<ParamsT>::ShaderBinding(const ProgramGroup::Ptr& program,
                                      const SbtRecordType& params) :
    ShaderBindingBase(program)
{
    this->add_dependency(program);
    this->optixObject_ = params;
}

template <typename ParamsT>
typename ShaderBinding<ParamsT>::Ptr ShaderBinding<ParamsT>::Create(
    const ProgramGroup::Ptr& program, const SbtRecordType& params)
{
    return Ptr(new ShaderBinding(program, params));
}

template <typename ParamsT>
void ShaderBinding<ParamsT>::do_build() const
{
    OPTIX_CHECK( optixSbtRecordPackHeader(*(this->program_), &this->optixObject_) );
}

template <typename ParamsT>
typename ShaderBinding<ParamsT>::SbtRecordType& ShaderBinding<ParamsT>::record()
{
    return *this;
}

template <typename ParamsT>
const typename ShaderBinding<ParamsT>::SbtRecordType& ShaderBinding<ParamsT>::record() const
{
    return *this;
}

template <typename ParamsT>
unsigned int ShaderBinding<ParamsT>::record_size() const
{
    return sizeof(SbtRecordType);
}

template <typename ParamsT>
void ShaderBinding<ParamsT>::fill_sbt_record(void* dst) const
{
    this->build();
    std::memcpy(dst, &this->optixObject_, this->record_size());
}

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_SHADER_BINDING_H_
