#include <optix_helpers/Buffer.h>

namespace optix_helpers {

BufferObj::BufferObj(const optix::Buffer& buffer, const std::string& name) :
    buffer_(buffer),
    name_(name)
{}

optix::Buffer BufferObj::buffer() const
{
    return buffer_;
}

BufferObj::operator optix::Buffer() const
{
    return buffer_;
}

optix::Buffer BufferObj::operator->() const
{
    return buffer_;
}

std::string BufferObj::name() const
{
    return name_;
}

}; //namespace optix_helpers

