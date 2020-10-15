#include <optix_helpers/Buffer.h>

namespace optix_helpers {

BufferObj::BufferObj(const Context& context, 
                     RTbuffertype bufferType,
                     RTformat format,
                     const std::string& name) :
    NamedObject<optix::Buffer>((*context)->createBuffer(bufferType, format), name)
{}

BufferObj::BufferObj(const optix::Buffer& buffer, const std::string& name) :
    NamedObject<optix::Buffer>(buffer, name)
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

BufferObj::Shape BufferObj::shape() const
{
    Shape res;
    this->buffer()->getSize(res.width, res.height);
    return res;
}

void BufferObj::unmap() const
{
    object_->unmap();
}

//Buffer::Buffer()
//{}
//
//Buffer::Buffer(const Context& context,
//               RTbuffertype bufferType,
//               RTformat format,
//               const std::string& name) :
//    Handle<BufferObj>(context, bufferType, format, name)
//{
//}


RenderBufferObj::RenderBufferObj(const Context& context, RTformat format,
                                 const std::string& name) :
    BufferObj(context, RT_BUFFER_OUTPUT, format, name)
{}

//Buffer::Buffer(const std::shared_ptr<BufferObj>& obj) :
//    Handle<BufferObj>(obj)
//{}

RenderBufferObj::RenderBufferObj(const optix::Buffer& buffer, const std::string& name) :
    BufferObj(buffer, name)
{}

//RenderBuffer::RenderBuffer()
//{}
//
//RenderBuffer::RenderBuffer(const Context& context, RTformat format,
//                           const std::string& name) :
//    Handle<RenderBufferObj>(context, format, name)
//{}
//
//RenderBuffer::RenderBuffer(const std::shared_ptr<RenderBufferObj>& obj) :
//    Handle<RenderBufferObj>(obj)
//{}
//
//RenderBuffer::operator Buffer()
//{
//    return Buffer(std::dynamic_pointer_cast<BufferObj>(this->obj_));
//}

}; //namespace optix_helpers

