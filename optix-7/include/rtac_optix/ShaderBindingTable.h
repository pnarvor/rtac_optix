#ifndef _DEF_RTAC_OPTIX_SHADER_BINDING_TABLE_H_
#define _DEF_RTAC_OPTIX_SHADER_BINDING_TABLE_H_

#include <iostream>
#include <vector>
#include <unordered_map>

#include <rtac_base/cuda/DeviceVector.h>

#include <rtac_optix/Instance.h>
#include <rtac_optix/ObjectInstance.h>
#include <rtac_optix/GroupInstance.h>

namespace rtac { namespace optix {

template <unsigned int RaytypeCountV>
class ShaderBindingTable
{
    public:

    static constexpr unsigned int RaytypeCount = RaytypeCountV;
    using Buffer = cuda::DeviceVector<uint8_t>;

    protected:

    using MaterialRecordsIndexes = std::unordered_map<MaterialBase::Ptr,
                                                      std::vector<unsigned int>>;
    
    std::vector<ObjectInstance::Ptr> objects_; // Instances which contains materials
    MaterialRecordsIndexes  materials_;
    unsigned int            hitRecordsCount_;
    unsigned int            hitRecordsSize_;
    Buffer                  hitRecordsData_;
    
    //void compute_offsets(const Instance::Ptr& instance);
    void add_material_record_index(const MaterialBase::Ptr& material, unsigned int index);

    public:

    ShaderBindingTable();

    void add_object(const ObjectInstance::Ptr& object);

    void fill_sbt(OptixShaderBindingTable& sbt);
};

template <unsigned int RaytypeCountV>
//ShaderBindingTable<RaytypeCountV>::ShaderBindingTable(const Instance::Ptr& topObject) :
ShaderBindingTable<RaytypeCountV>::ShaderBindingTable() :
    objects_(0),
    hitRecordsCount_(0),
    hitRecordsSize_(0)
{
    //this->compute_offsets(topObject);
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::add_material_record_index(
    const MaterialBase::Ptr& material, unsigned int index)
{
    if(materials_.find(material) == materials_.end()) {
        materials_[material] = std::vector<unsigned int>();
    }
    materials_[material].push_back(RaytypeCount*index + material->raytype_index());
    hitRecordsSize_ = std::max(hitRecordsSize_, material->record_size());
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::add_object(const ObjectInstance::Ptr& object)
{
    if(std::find(objects_.begin(), objects_.end(), object) != objects_.end()) {
        // object already added.
        return;
    }

    objects_.push_back(object);
    object->set_sbt_offset(hitRecordsCount_ * RaytypeCount);
    std::cout << "Object sbt offset : " <<  hitRecordsCount_ * RaytypeCount << std::endl;
    for(int i = 0; i < object->material_count(); i++) {
        auto mat = object->material(i);
        if(mat)
            this->add_material_record_index(mat, hitRecordsCount_ + i);
    }
    hitRecordsCount_ += object->material_count();
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::fill_sbt(OptixShaderBindingTable& sbt)
{
    std::vector<uint8_t> recordsData(RaytypeCountV * hitRecordsCount_ * hitRecordsSize_, 0);
    for(auto& item : materials_) {
        for(auto offset : item.second) {
            std::cout << "Filling sbt " << offset*hitRecordsSize_ << std::endl;
            item.first->fill_sbt_record(recordsData.data() + offset*hitRecordsSize_);
        }
    }
    hitRecordsData_ = recordsData;

    std::cout << RaytypeCount     << std::endl;
    std::cout << hitRecordsCount_ << std::endl;
    std::cout << hitRecordsSize_  << std::endl;


    sbt.hitgroupRecordBase          = (CUdeviceptr)hitRecordsData_.data();
    sbt.hitgroupRecordStrideInBytes = hitRecordsSize_;
    sbt.hitgroupRecordCount         = RaytypeCountV * hitRecordsCount_;
}

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_SHADER_BINDING_TABLE_H_
