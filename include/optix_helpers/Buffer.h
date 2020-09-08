#ifndef _DEF_OPTIX_HELPERS_BUFFER_H_
#define _DEF_OPTIX_HELPERS_BUFFER_H_

#include <iostream>

#include <optixu/optixpp.h>

namespace optix_helpers {

class Buffer
{
    protected:

    optix::Buffer buffer_;
    std::string name_;

    public:
    
    Buffer(const optix::Buffer& buffer, const std::string& name);

    optix::Buffer operator->();
    optix::Buffer operator->() const;
    optix::Buffer buffer() const;
    std::string name() const;
};

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_BUFFER_H_
