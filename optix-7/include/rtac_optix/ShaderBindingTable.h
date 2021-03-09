#ifndef _DEF_RTAC_OPTIX_SHADER_BINDING_TABLE_H_
#define _DEF_RTAC_OPTIX_SHADER_BINDING_TABLE_H_

#include <iostream>
#include <vector>
#include <unordered_map>
#include <algorithm>

#include <rtac_base/cuda/DeviceVector.h>

#include <rtac_optix/OptixWrapper.h>
#include <rtac_optix/ShaderBinding.h>
#include <rtac_optix/Instance.h>
#include <rtac_optix/ObjectInstance.h>
#include <rtac_optix/GroupInstance.h>

namespace rtac { namespace optix {

template <unsigned int RaytypeCountV>
class ShaderBindingTable : OptixWrapper<OptixShaderBindingTable>
{
    public:

    using Ptr      = OptixWrapperHandle<ShaderBindingTable<RaytypeCountV>>;
    using ConstPtr = OptixWrapperHandle<const ShaderBindingTable<RaytypeCountV>>;

    static constexpr unsigned int RaytypeCount = RaytypeCountV;
    using Buffer = cuda::DeviceVector<uint8_t>;

    protected:
    
    using MissRecords            = std::vector<MaterialBase::Ptr>;
    using MaterialRecordsIndexes = std::unordered_map<MaterialBase::Ptr,
                                                      std::vector<unsigned int>,
                                                      MaterialBase::Ptr::Hash>;
    
    ShaderBindingBase::Ptr raygenRecord_;
    mutable Buffer         raygenRecordData_;

    ShaderBindingBase::Ptr exceptionRecord_;
    mutable Buffer         exceptionRecordData_;

    MissRecords    missRecords_;
    mutable Buffer missRecordsData_;

    std::vector<ObjectInstance::Ptr> objects_; // Instances which contains materials
    MaterialRecordsIndexes           materials_;
    unsigned int                     hitRecordsCount_;
    unsigned int                     hitRecordsSize_;
    mutable Buffer                   hitRecordsData_;
    
    void add_material_record_index(const MaterialBase::Ptr& material, unsigned int index);

    void do_build() const;
    void fill_raygen_record() const;
    void fill_exception_record() const;
    void fill_miss_records() const;
    void fill_hit_records() const;

    ShaderBindingTable();

    public:

    static Ptr Create();

    const OptixShaderBindingTable* sbt() const;
    
    void set_raygen_record(const ShaderBindingBase::Ptr& record);
    void set_exception_record(const ShaderBindingBase::Ptr& record);
    void add_miss_record(const MaterialBase::Ptr& record);
    void add_object(const ObjectInstance::Ptr& object);
};

template <unsigned int RaytypeCountV>
ShaderBindingTable<RaytypeCountV>::ShaderBindingTable() :
    missRecords_(RaytypeCount, nullptr),
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
void ShaderBindingTable<RaytypeCountV>::do_build() const
{
    this->fill_raygen_record();
    this->fill_exception_record();
    this->fill_miss_records();
    this->fill_hit_records();
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::fill_raygen_record() const
{
    if(!raygenRecord_) {
        throw std::runtime_error(
            "No raygen record set. Cannot instanciate ShaderBindingTable");
    }
    std::vector<uint8_t> tmp(raygenRecord_->record_size());
    raygenRecord_->fill_sbt_record(tmp.data());
    raygenRecordData_ = tmp;
    optixObject_.raygenRecord = (CUdeviceptr)raygenRecordData_.data();
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::fill_exception_record() const
{
    if(!exceptionRecord_)
        return;

    std::vector<uint8_t> tmp(exceptionRecord_->record_size());
    exceptionRecord_->fill_sbt_record(tmp.data());
    exceptionRecordData_ = tmp;
    optixObject_.exceptionRecord = (CUdeviceptr)exceptionRecordData_.data();
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::fill_miss_records() const
{
    unsigned int recordSize = 0;
    for(auto mat : missRecords_) {
        if(mat)
            recordSize = std::max(recordSize, mat->record_size());
    }

    if(!recordSize)
        return;

    std::vector<uint8_t> tmp(RaytypeCount * recordSize, 0);
    uint8_t* data = tmp.data();
    for(auto mat : missRecords_) {
        if(mat)
            mat->fill_sbt_record(data);
        data += recordSize;
    }
    missRecordsData_ = tmp;
    optixObject_.missRecordBase          = (CUdeviceptr)missRecordsData_.data();
    optixObject_.missRecordStrideInBytes = recordSize;
    optixObject_.missRecordCount         = RaytypeCount;
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::fill_hit_records() const
{
    std::vector<uint8_t> recordsData(RaytypeCountV * hitRecordsCount_ * hitRecordsSize_, 0);
    for(auto& item : materials_) {
        for(auto offset : item.second) {
            item.first->fill_sbt_record(recordsData.data() + offset*hitRecordsSize_);
        }
    }
    hitRecordsData_ = recordsData;

    optixObject_.hitgroupRecordBase          = (CUdeviceptr)hitRecordsData_.data();
    optixObject_.hitgroupRecordStrideInBytes = hitRecordsSize_;
    optixObject_.hitgroupRecordCount         = RaytypeCountV * hitRecordsCount_;
}

template <unsigned int RaytypeCountV>
const OptixShaderBindingTable* ShaderBindingTable<RaytypeCountV>::sbt() const
{
    this->build();
    return &optixObject_;
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::set_raygen_record(const ShaderBindingBase::Ptr& record)
{
    raygenRecord_ = record;
    this->add_dependency(record);
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::set_exception_record(const ShaderBindingBase::Ptr& record)
{
    exceptionRecord_ = record;
    this->add_dependency(record);
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::add_miss_record(const MaterialBase::Ptr& record)
{
    if(record->raytype_index() >= RaytypeCount) {
        throw std::runtime_error("In valid miss record raytype index.");
    }
    missRecords_[record->raytype_index()] = record;
    this->add_dependency(record);
}

template <unsigned int RaytypeCountV>
void ShaderBindingTable<RaytypeCountV>::add_material_record_index(
    const MaterialBase::Ptr& material, unsigned int index)
{
    if(materials_.find(material) == materials_.end()) {
        materials_[material] = std::vector<unsigned int>();
        this->add_dependency(material);
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
    objects_.back()->set_sbt_offset(hitRecordsCount_ * RaytypeCount);
    for(auto mat : objects_.back()->materials()) {
        if(mat.second)
            this->add_material_record_index(mat.second, hitRecordsCount_ + mat.first);
    }
    hitRecordsCount_ += object->material_count();
}

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_SHADER_BINDING_TABLE_H_
