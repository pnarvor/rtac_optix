#include <rtac_optix/Material.h>

namespace rtac { namespace optix {

MaterialBase::MaterialBase(unsigned int rayTypeIndex, const ProgramGroup::Ptr& hitPrograms) :
    rayTypeIndex_(rayTypeIndex),
    hitPrograms_(hitPrograms),
    needsUpdate_(true)
{}

unsigned int MaterialBase::raytype_index() const
{
    return rayTypeIndex_;
}

ProgramGroup::Ptr MaterialBase::hit_programs()
{
    return hitPrograms_;
}

ProgramGroup::ConstPtr MaterialBase::hit_programs() const
{
    return hitPrograms_;
}

bool MaterialBase::needs_update() const
{
    return needsUpdate_;
}

}; //namespace optix
}; //namespace rtac
