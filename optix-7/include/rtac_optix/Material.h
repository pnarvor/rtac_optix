#ifndef _DEF_RTAC_OPTIX_MATERIAL_H_
#define _DEF_RTAC_OPTIX_MATERIAL_H_

#include <iostream>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>
#include <rtac_optix/ProgramGroup.h>

namespace rtac { namespace optix {

class MaterialBase
{
    // Generic base Material class to be used in Instance type, but must not be
    // instanciated on its own.
    public:

    using Ptr      = Handle<MaterialBase>;
    using ConstPtr = Handle<const MaterialBase>;

    protected:

    unsigned int              rayTypeIndex_;
    // putting these mutable for now. To be fixed.
    mutable ProgramGroup::Ptr hitPrograms_; // Must have been instanciated with a Pipeline.
    mutable bool              needsUpdate_;
    
    // have to be instanciated by a Material<> type
    MaterialBase(unsigned int rayTypeIndex, const ProgramGroup::Ptr& hitPrograms);

    public:

    unsigned int raytype_index() const;
    bool needs_update() const;

    ProgramGroup::Ptr      hit_programs();
    ProgramGroup::ConstPtr hit_programs() const;
    
    // Declaring this function to make the class abstract.
    virtual void fill_sbt_record(void* dst) const = 0;
};

template <typename RayT, class ParamsT>
class Material : public MaterialBase
{
    public:

    using RayType       = RayT;
    using ParamsType    = ParamsT;
    using SbtRecordType = SbtRecord<ParamsT>;

    using Ptr      = Handle<Material<RayType, ParamsType>>;
    using ConstPtr = Handle<const Material<RayType, ParamsType>>;
    
    protected:

    mutable SbtRecordType sbtRecord_;

    Material(const ProgramGroup::Ptr& hitPrograms, const ParamsType& params);

    public:

    static Ptr Create(const ProgramGroup::Ptr& hitPrograms,
                      const ParamsType& params = zero<ParamsType>());
    
    ParamsType& params();
    const ParamsType& params() const;

    void fill_sbt_record(void* dst) const;
};

template <typename RayT, class ParamsT>
Material<RayT,ParamsT>::Material(const ProgramGroup::Ptr& hitPrograms,
                                 const ParamsType& params) :
    MaterialBase(RayT::Index, hitPrograms),
    sbtRecord_(params)
{}

template <typename RayT, class ParamsT>
typename Material<RayT,ParamsT>::Ptr Material<RayT,ParamsT>::Create(
    const ProgramGroup::Ptr& hitPrograms, const ParamsType& params)
{
    return Ptr(new Material<RayT,ParamsT>(hitPrograms, params));
}

template <typename RayT, class ParamsT>
ParamsT& Material<RayT,ParamsT>::params()
{
    this->needsUpdate_ = true;
    return sbtRecord_.data;
}

template <typename RayT, class ParamsT>
const ParamsT& Material<RayT,ParamsT>::params() const
{
    return sbtRecord_.data;
}

template <typename RayT, class ParamsT>
void Material<RayT,ParamsT>::fill_sbt_record(void* dst) const
{
    if(this->needsUpdate_) {
        OPTIX_CHECK( optixSbtRecordPackHeader(*(this->hitPrograms_), &sbtRecord_) );
        this->needsUpdate_ = false;
    }
    
    std::memcpy(dst, &sbtRecord_, sizeof(sbtRecord_));
}


}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_MATERIAL_H_




