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

}; //namespace optix_helpers

