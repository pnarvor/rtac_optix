#include <optix_helpers/Buffer.h>

namespace optix_helpers {

BufferObj::BufferObj(const Context& context, 
                     RTbuffertype bufferType,
                     RTformat format,
                     const std::string& name) :
    NamedObject<optix::Buffer>((*context)->createBuffer(bufferType, format), name)
{}

const optix::Buffer BufferObj::buffer() const
{
    return object_;
}

void BufferObj::set_size(size_t width, size_t height)
{
    object_->setSize(width, height);
}

optix::Buffer BufferObj::buffer()
{
    return object_;
}

RenderBufferObj::RenderBufferObj(const Context& context, RTformat format,
                                 const std::string& name) :
    BufferObj(context, RT_BUFFER_OUTPUT, format, name)
{}

RenderBuffer::RenderBuffer(const Context& context, RTformat format,
                           const std::string& name) :
    Handle<RenderBufferObj>(context, format, name)
{
}

RenderBuffer::operator Buffer()
{
    return Buffer(std::dynamic_pointer_cast<BufferObj>(this->obj_));
}

}; //namespace optix_helpers

