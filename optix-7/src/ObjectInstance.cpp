#include <rtac_optix/ObjectInstance.h>

namespace rtac { namespace optix {

ObjectInstance::ObjectInstance(const GeometryAccelStruct::Ptr& geometry,
                               unsigned int instanceId) :
    Instance(geometry, instanceId),
    materials_(0)
{}

ObjectInstance::Ptr ObjectInstance::Create(const GeometryAccelStruct::Ptr& geometry,
                                           unsigned int instanceId)
{
    return Ptr(new ObjectInstance(geometry, instanceId));
}

unsigned int ObjectInstance::material_count() const
{
    return this->child_->sbt_width();
}

MaterialBase::ConstPtr ObjectInstance::material(unsigned int index) const
{
    if(index >= this->material_count()) {
        std::ostringstream oss;
        oss << "Invalid material index value (index is " << index
            << ", geometry has " << this->material_count() << ")";
        throw std::out_of_range(oss.str());
    }

    if(index >= materials_.size()) {
        // no error, we would not have any material anyway.
        return nullptr;
    }
    
    return materials_[index];
}

MaterialBase::Ptr ObjectInstance::material(unsigned int index)
{
    if(index >= this->material_count()) {
        std::ostringstream oss;
        oss << "Invalid material index value (index is " << index
            << ", geometry has " << this->material_count() << ")";
        throw std::out_of_range(oss.str());
    }

    if(index >= materials_.size()) {
        // no error, we would not have any material anyway.
        return nullptr;
    }
    
    return materials_[index];
}

void ObjectInstance::set_material(const MaterialBase::Ptr& material, unsigned int index)
{
    if(index >= this->material_count()) {
        std::ostringstream oss;
        oss << "Invalid material index value (index is " << index
            << ", geometry has " << this->material_count() << ")";
        throw std::out_of_range(oss.str());
    }

    if(materials_.size() <= this->material_count())
        materials_.resize(this->material_count());

    materials_[index] = material;
}

void ObjectInstance::unset_material(unsigned int index)
{
    if(index >= materials_.size()) {
        // no error, we would have deleted anyway.
        return;
    }
    materials_[index] = nullptr;
}

}; //namespace optix
}; //namespace rtac
