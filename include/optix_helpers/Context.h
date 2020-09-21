#ifndef _DEF_OPTIX_HELPERS_CONTEXT_H_
#define _DEF_OPTIX_HELPERS_CONTEXT_H_

#include <iostream>
#include <memory>

#include <optixu/optixpp.h>

#include <rtac_base/types/Mesh.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Nvrtc.h>
#include <optix_helpers/Source.h>
#include <optix_helpers/Program.h>
#include <optix_helpers/Buffer.h>
#include <optix_helpers/RayType.h>
//#include <optix_helpers/RayGenerator.h>

namespace optix_helpers {

class ContextObj
{
    protected:
    
    // Fix the mutable keyword use
    mutable optix::Context context_;
    mutable Nvrtc nvrtc_;

    public:

    ContextObj(int entryPointCount = 1);

    Program create_program(const Source& source,
                           const Sources& additionalHeaders = Sources()) const; 
    Buffer create_buffer(RTbuffertype bufferType, RTformat format, 
                         const std::string& name = "buffer") const;
    Buffer create_gl_buffer(RTbuffertype bufferType, RTformat format,
                            unsigned int glboId, const std::string& name = "buffer") const;
    RayType  create_raytype(const Source& rayDefinition) const;

    optix::Handle<optix::VariableObj> operator[](const std::string& varname);
    operator optix::Context()   const;
    optix::Context operator->();
    optix::Context operator->() const;
    optix::Context context()    const; //? should be const ?
};
using Context = Handle<ContextObj>;

}; // namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_CONTEXT_H_
