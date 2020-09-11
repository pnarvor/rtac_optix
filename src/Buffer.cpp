#include <optix_helpers/Buffer.h>

namespace optix_helpers {

BufferObj::BufferObj(const optix::Buffer& buffer, const std::string& name) :
    NamedObject<optix::Buffer>(buffer, name)
{}

const optix::Buffer BufferObj::buffer() const
{
    return object_;
}

optix::Buffer BufferObj::buffer()
{
    return object_;
}

}; //namespace optix_helpers

