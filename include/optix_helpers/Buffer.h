#ifndef _DEF_OPTIX_HELPERS_BUFFER_H_
#define _DEF_OPTIX_HELPERS_BUFFER_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/NamedObject.h>
#include <optix_helpers/Handle.h>

namespace optix_helpers {

class BufferObj : public NamedObject<optix::Buffer>
{
    public:
    
    BufferObj(const optix::Buffer& buffer, const std::string& name);

    optix::Buffer       buffer();
    const optix::Buffer buffer() const;
};
using Buffer = Handle<BufferObj>;

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_BUFFER_H_
