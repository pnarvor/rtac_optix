#ifndef _DEF_RTAC_OPTIX_SHADER_BINDING_H_
#define _DEF_RTAC_OPTIX_SHADER_BINDING_H_

#include <iostream>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>
#include <rtac_optix/ProgramGroup.h>

namespace rtac { namespace optix {

class ShaderBindingBase
{
    public:

    using Ptr      = Handle<ShaderBindingBase>;
    using ConstPtr = Handle<const ShaderBindingBase>;

    protected:

    // putting these mutable for now. To be fixed.
    mutable ProgramGroup::Ptr program_;
    mutable bool              needsUpdate_;

    ShaderBindingBase(const ProgramGroup::Ptr& program);
    
    public:

    bool needs_update() const;

    ProgramGroup::Ptr      program();
    ProgramGroup::ConstPtr program() const;

    virtual unsigned int record_size() const = 0;
    virtual void fill_sbt_record(void* dst) const = 0;
};

template <typename ParamsT>
class ShaderBinding : public virtual ShaderBindingBase
{
    public:

    using ParamsType    = ParamsT;
    using SbtRecordType = SbtRecord<ParamsT>;

    using Ptr      = Handle<ShaderBinding<ParamsType>>;
    using ConstPtr = Handle<const ShaderBinding<ParamsType>>;
    
    protected:

    mutable SbtRecordType sbtRecord_;

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
    ShaderBindingBase(program),
    sbtRecord_(params)
{}

template <typename ParamsT>
typename ShaderBinding<ParamsT>::Ptr ShaderBinding<ParamsT>::Create(
    const ProgramGroup::Ptr& program, const SbtRecordType& params)
{
    return Ptr(new ShaderBinding(program, params));
}

template <typename ParamsT>
typename ShaderBinding<ParamsT>::SbtRecordType& ShaderBinding<ParamsT>::record()
{
    return sbtRecord_;
}

template <typename ParamsT>
const typename ShaderBinding<ParamsT>::SbtRecordType& ShaderBinding<ParamsT>::record() const
{
    return sbtRecord_;
}

template <typename ParamsT>
unsigned int ShaderBinding<ParamsT>::record_size() const
{
    return sizeof(SbtRecordType);
}

template <typename ParamsT>
void ShaderBinding<ParamsT>::fill_sbt_record(void* dst) const
{
    if(this->needsUpdate_) {
        OPTIX_CHECK( optixSbtRecordPackHeader(*(this->program_), &sbtRecord_) );
        this->needsUpdate_ = false;
    }
    std::memcpy(dst, &sbtRecord_, this->record_size());
}

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_SHADER_BINDING_H_
