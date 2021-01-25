#ifndef _DEF_OPTIX_HELPERS_BUFFER_H_
#define _DEF_OPTIX_HELPERS_BUFFER_H_

#include <iostream>
#include <cassert>

#include <optixu/optixpp.h>

#include <rtac_base/types/Shape.h>
#include <rtac_base/cuda/utils.h>
#include <rtac_base/cuda/DeviceVector.h>

#include <optix_helpers/NamedObject.h>
#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>

namespace optix_helpers {

class Buffer : public NamedObject<optix::Buffer>
{
    public:

    using Ptr      = Handle<Buffer>;
    using ConstPtr = Handle<const Buffer>;
    using Shape    = rtac::types::Shape<size_t>;

    protected:
    //public:

    // only for subclasses
    Buffer(const optix::Buffer& buffer, const std::string& name);

    void*       device_pointer(int deviceOrdinal = 0);
    const void* device_pointer(int deviceOrdinal = 0) const;

    public:
    
    static Ptr New(const Context::ConstPtr& context,
                   RTbuffertype bufferType,
                   RTformat format,
                   const std::string& name);
    Buffer(const Context::ConstPtr& context,
           RTbuffertype bufferType,
           RTformat format,
           const std::string& name);

    virtual void set_size(size_t size);
    virtual void set_size(size_t width, size_t height);

    optix::Buffer        buffer();
    const optix::Buffer& buffer() const;

    Shape shape() const;
    size_t size() const;
    RTsize element_size() const;

    template <typename T>
    rtac::cuda::DeviceVector<T>& to_device_vector(rtac::cuda::DeviceVector<T>& out,
                                                    int deviceOrdinal = 0) const;
    template <typename T>
    rtac::cuda::DeviceVector<T> to_device_vector(int deviceOrdinal = 0) const;

    template <typename T>
    T map(unsigned int mapFlags = RT_BUFFER_MAP_WRITE, T* userOutput = NULL);
    template <typename T>
    T map(T* userOutput = NULL) const;
    void unmap() const;
};

template <typename T>
rtac::cuda::DeviceVector<T>& Buffer::to_device_vector(rtac::cuda::DeviceVector<T>& out, 
                                                        int deviceOrdinal) const
{
    // not really clean, find a better way (templated Buffer ?)
    assert(this->size() * this->buffer()->getElementSize() == out.size() * sizeof(T));
    int deviceOrdinalCuda;
    rtDeviceGetAttribute(deviceOrdinal, RT_DEVICE_ATTRIBUTE_CUDA_DEVICE_ORDINAL,
                         sizeof(int), &deviceOrdinalCuda);
    rtac::cuda::set_device(deviceOrdinalCuda);
    rtac::cuda::memcpy::copy_device_to_device(reinterpret_cast<void*>(out.data()),
                                                this->device_pointer(deviceOrdinal),
                                                out.size() * sizeof(T));
    return out;
}
template <typename T>
rtac::cuda::DeviceVector<T> Buffer::to_device_vector(int deviceOrdinal) const
{
    // not really clean, find a better way (templated Buffer ?)
    assert((this->size() * this->buffer()->getElementSize())  % sizeof(T) == 0);
    
    // allocating
    rtac::cuda::DeviceVector<T> res((this->size() * this->buffer()->getElementSize()) / sizeof(T));

    this->to_device_vector(res, deviceOrdinal);

    return res;
}

template <typename T>
T Buffer::map(unsigned int mapFlags, T* userOutput)
{
    return static_cast<T>(object_->map(0, mapFlags, userOutput));
}

template <typename T>
T Buffer::map(T* userOutput) const
{
    return static_cast<T>(object_->map(0, RT_BUFFER_MAP_READ, userOutput));
}

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_BUFFER_H_
