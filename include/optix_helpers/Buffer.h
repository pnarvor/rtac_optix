#ifndef _DEF_OPTIX_HELPERS_BUFFER_H_
#define _DEF_OPTIX_HELPERS_BUFFER_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <rtac_base/types/Shape.h>

#include <optix_helpers/NamedObject.h>
#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>

namespace optix_helpers {

class BufferObj : public NamedObject<optix::Buffer>
{
    public:

    using Shape = rtac::types::Shape<size_t>;

    protected:

    // only for subclasses
    BufferObj(const optix::Buffer& buffer, const std::string& name);

    public:
    
    BufferObj(const Context& context,
              RTbuffertype bufferType,
              RTformat format,
              const std::string& name);

    virtual void set_size(size_t width, size_t height);

    optix::Buffer       buffer();
    const optix::Buffer buffer() const;

    Shape shape() const;
};

class Buffer : public Handle<BufferObj>
{
    public:

    using Shape = BufferObj::Shape;

    Buffer();
    Buffer(const Context& context,
           RTbuffertype bufferType,
           RTformat format,
           const std::string& name);

    // for downcasting
    Buffer(const std::shared_ptr<BufferObj>& obj);
};

class RenderBufferObj : public BufferObj
{
    protected:
    
    // only for subclasses
    RenderBufferObj(const optix::Buffer& buffer, const std::string& name);

    public:
    
    RenderBufferObj(const Context& context, RTformat format,
                    const std::string& name);
};

class RenderBuffer : public Handle<RenderBufferObj>
{
    public:
    
    RenderBuffer();
    RenderBuffer(const Context& context, RTformat format,
                 const std::string& name);
    RenderBuffer(const std::shared_ptr<RenderBufferObj>& obj);

    operator Buffer();    
};

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_BUFFER_H_
