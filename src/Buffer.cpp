#include <optix_helpers/Buffer.h>

namespace optix_helpers {

Buffer::Buffer(const optix::Buffer& buffer, const std::string& name) :
    buffer_(buffer),
    name_(name)
{}

optix::Buffer Buffer::operator->()
{
    return buffer_;
}

optix::Buffer Buffer::operator->() const
{
    return buffer_;
}

optix::Buffer Buffer::buffer() const
{
    return buffer_;
}

std::string Buffer::name() const
{
    return name_;
}

}; //namespace optix_helpers

