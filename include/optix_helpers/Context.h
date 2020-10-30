#ifndef _DEF_OPTIX_HELPERS_CONTEXT_H_
#define _DEF_OPTIX_HELPERS_CONTEXT_H_

#include <iostream>
#include <memory>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Nvrtc.h>
#include <optix_helpers/Source.h>
#include <optix_helpers/Program.h>
#include <optix_helpers/RayType.h>

namespace optix_helpers {

class Context
{
    public:
    
    using Ptr      = Handle<Context>;
    using ConstPtr = Handle<const Context>;

    protected:
    
    // Fix the mutable keyword use
    mutable optix::Context context_;
    mutable Nvrtc nvrtc_;

    public:

    static Ptr New(int entryPointCount = 1);
    Context(int entryPointCount = 1);

    Program::Ptr create_program(const Source::ConstPtr& source,
                                const Sources& additionalHeaders = Sources()) const; 
    template <typename RayT>
    RayType instanciate_raytype() const;

    optix::Handle<optix::VariableObj> operator[](const std::string& varname);
    operator optix::Context()   const;
    optix::Context operator->();
    optix::Context operator->() const;
    optix::Context context()    const; //? should be const ?
};

// Implementation
template <typename RayT>
RayType Context::instanciate_raytype() const
{
    if(RayT::index == RayType::uninitialized) {
        // RayT never instanciated. Assigning new index.
        RayT::index  = context_->getRayTypeCount();
        context_->setRayTypeCount(RayT::index + 1);
    }
    return RayType(RayT::index, RayT::definition);
}

}; // namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_CONTEXT_H_
