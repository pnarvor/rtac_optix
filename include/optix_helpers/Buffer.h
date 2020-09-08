#ifndef _DEF_OPTIX_HELPERS_BUFFER_H_
#define _DEF_OPTIX_HELPERS_BUFFER_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>

namespace optix_helpers {

class BufferObj
{
    protected:

    optix::Buffer buffer_;
    std::string name_;

    public:
    
    BufferObj(const optix::Buffer& buffer, const std::string& name);

    optix::Buffer buffer() const;
    operator optix::Buffer() const;
    optix::Buffer operator->() const;
    std::string name() const;
};
using Buffer = Handle<BufferObj>;

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_BUFFER_H_
