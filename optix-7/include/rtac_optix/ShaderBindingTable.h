#ifndef _DEF_RTAC_OPTIX_SHADER_BINDING_TABLE_H_
#define _DEF_RTAC_OPTIX_SHADER_BINDING_TABLE_H_

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include <rtac_base/cuda/DeviceVector.h>

#include <rtac_optix/ShaderBinding.h>
#include <rtac_optix/Instance.h>
#include <rtac_optix/ObjectInstance.h>
#include <rtac_optix/GroupInstance.h>

namespace rtac { namespace optix {

template <unsigned int RaytypeCountV>
class ShaderBindingTable
{
    public:

    using Ptr      = Handle<ShaderBindingTable<RaytypeCountV>>;
    using ConstPtr = Handle<const ShaderBindingTable<RaytypeCountV>>;

    static constexpr unsigned int RaytypeCount = RaytypeCountV;
    using Buffer = cuda::DeviceVector<uint8_t>;

    protected:
    
    using MissRecords            = std::vector<MaterialBase::Ptr>;
    using MaterialRecordsIndexes = std::unordered_map<MaterialBase::Ptr,
                                                      std::vector<unsigned int>>;
    
    OptixShaderBindingTable sbt_;
    
    ShaderBindingBase::Ptr raygenData_;
    Buffer                 raygenRecord_;

    ShaderBindingBase::Ptr exceptionData_;
    Buffer                 exceptionRecord_;

    MissRecords missRecordsData_;
    Buffer      missRecords_;

    std::vector<ObjectInstance::Ptr> objects_; // Instances which contains materials
    MaterialRecordsIndexes           materials_;
    unsigned int                     hitRecordsCount_;
    unsigned int                     hitRecordsSize_;
    Buffer                           hitRecords_;
    
    void add_material_record_index(const MaterialBase::Ptr& material, unsigned int index);

    void fill_raygen_record();
    void fill_exception_record();
    void fill_miss_records();
    void fill_hit_records();

    public:

    ShaderBindingTable();
    static Ptr Create();

    const OptixShaderBindingTable* sbt();
    operator OptixShaderBindingTable();
    
    void set_raygen_record(const ShaderBindingBase::Ptr& record);
    void set_exception_record(const ShaderBindingBase::Ptr& record);
    void add_miss_record(const MaterialBase::Ptr& record);
    void add_object(const ObjectInstance::Ptr& object);
};

template <unsigned int RaytypeCountV>
ShaderBindingTable<RaytypeCountV>::ShaderBindingTable() :
    sbt_(zero<OptixShaderBindingTable>()),
    missRecordsData_(RaytypeCount, nullptr),
    objects_(0),
    hitRecordsCount_(0),
    hitRecordsSize_(0)
{}

template <unsigned int RaytypeCountV>
typename ShaderBindingTable<RaytypeCountV>::Ptr ShaderBindingTable<RaytypeCountV>::Create()
{
    return Ptr(new ShaderBindingTable<RaytypeCountV>());
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
const OptixShaderBindingTable* ShaderBindingTable<RaytypeCountV>::sbt()
{
    this->fill_raygen_record();
    this->fill_exception_record();
    this->fill_miss_records();
    this->fill_hit_records();
    return &sbt_;
}

template <unsigned int RaytypeCountV>
ShaderBindingTable<RaytypeCountV>::operator OptixShaderBindingTable()
{
    return *(this->sbt());
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::set_raygen_record(const ShaderBindingBase::Ptr& record)
{
    raygenData_ = record;
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::fill_raygen_record()
{
    if(!raygenData_) {
        throw std::runtime_error(
            "Not raygen record set. Cannot instanciate ShaderBindingTable");
    }
    std::vector<uint8_t> tmp(raygenData_->record_size());
    raygenData_->fill_sbt_record(tmp.data());
    raygenRecord_ = tmp;
    sbt_.raygenRecord = (CUdeviceptr)raygenRecord_.data();
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::set_exception_record(const ShaderBindingBase::Ptr& record)
{
    exceptionData_ = record;
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::fill_exception_record()
{
    if(!exceptionData_)
        return;

    std::vector<uint8_t> tmp(exceptionData_->record_size());
    exceptionData_->fill_sbt_record(tmp.data());
    exceptionRecord_ = tmp;
    sbt_.exceptionRecord = (CUdeviceptr)exceptionRecord_.data();
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::add_miss_record(const MaterialBase::Ptr& record)
{
    if(record->raytype_index() >= RaytypeCount) {
        throw std::runtime_error("In valid miss record raytype index.");
    }
    missRecordsData_[record->raytype_index()] = record;
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::fill_miss_records()
{
    unsigned int recordSize = 0;
    for(auto mat : missRecordsData_) {
        if(mat)
            recordSize = std::max(recordSize, mat->record_size());
    }

    if(!recordSize)
        return;

    std::vector<uint8_t> tmp(RaytypeCount * recordSize, 0);
    uint8_t* data = tmp.data();
    for(auto mat : missRecordsData_) {
        if(mat)
            mat->fill_sbt_record(data);
        data += recordSize;
    }
    missRecords_ = tmp;
    sbt_.missRecordBase          = (CUdeviceptr)missRecords_.data();
    sbt_.missRecordStrideInBytes = recordSize;
    sbt_.missRecordCount         = RaytypeCount;
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::add_object(const ObjectInstance::Ptr& object)
{
    if(std::find(objects_.begin(), objects_.end(), object) != objects_.end()) {
        // object already added.
        return;
    }

    objects_.push_back(object);
    objects_.back()->set_sbt_offset(hitRecordsCount_ * RaytypeCount);
    for(auto mat : objects_.back()->materials()) {
        if(mat.second)
            this->add_material_record_index(mat.second, hitRecordsCount_ + mat.first);
    }
    hitRecordsCount_ += object->material_count();
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::fill_hit_records()
{
    std::vector<uint8_t> recordsData(RaytypeCountV * hitRecordsCount_ * hitRecordsSize_, 0);
    for(auto& item : materials_) {
        for(auto offset : item.second) {
            item.first->fill_sbt_record(recordsData.data() + offset*hitRecordsSize_);
        }
    }
    hitRecords_ = recordsData;

    sbt_.hitgroupRecordBase          = (CUdeviceptr)hitRecords_.data();
    sbt_.hitgroupRecordStrideInBytes = hitRecordsSize_;
    sbt_.hitgroupRecordCount         = RaytypeCountV * hitRecordsCount_;
}

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_SHADER_BINDING_TABLE_H_
