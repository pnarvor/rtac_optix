#ifndef _DEF_OPTIX_HELPERS_BUFFER_H_
#define _DEF_OPTIX_HELPERS_BUFFER_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <rtac_base/types/Shape.h>

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

    // only for subclasses
    Buffer(const optix::Buffer& buffer, const std::string& name);

    public:
    
    static Ptr New(const Context::ConstPtr& context,
                   RTbuffertype bufferType,
                   RTformat format,
                   const std::string& name);
    Buffer(const Context::ConstPtr& context,
           RTbuffertype bufferType,
           RTformat format,
           const std::string& name);

    virtual void set_size(size_t width, size_t height);

    optix::Buffer       buffer();
    const optix::Buffer buffer() const;

    Shape shape() const;

    template <typename T>
    T* map(unsigned int mapFlags, T* userOutput = NULL);
    template <typename T>
    const T* map(T* userOutput = NULL) const;
    void unmap() const;
};

template <typename T>
T* Buffer::map(unsigned int mapFlags, T* userOutput)
{
    return static_cast<T*>(object_->map(0, mapFlags, userOutput));
}

template <typename T>
const T* Buffer::map(T* userOutput) const
{
    return static_cast<const T*>(object_->map(0, RT_BUFFER_MAP_READ, userOutput));
}


}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_BUFFER_H_
