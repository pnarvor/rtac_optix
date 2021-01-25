#ifndef _DEF_DEFINE_OPTIX_HELPERS_GEOMETRY_H_
#define _DEF_DEFINE_OPTIX_HELPERS_GEOMETRY_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/Program.h>

namespace optix_helpers {

class Geometry
{
    public:

    using Ptr      = Handle<Geometry>;
    using ConstPtr = Handle<const Geometry>;

    protected:
    
    optix::Geometry geometry_;
    Program::Ptr intersectionProgram_;
    Program::Ptr boundingboxProgram_;

    public:

    static Ptr New(const Context::ConstPtr& context,
                   const Program::Ptr& intersectionProgram,
                   const Program::Ptr& boundingboxProgram,
                   size_t primitiveCount);

    Geometry(const Context::ConstPtr& context,
             const Program::Ptr& intersectionProgram,
             const Program::Ptr& boundingboxProgram,
             size_t primitiveCount);
    
    Program::Ptr intersection_program();
    Program::Ptr boundingbox_program();

    Program::ConstPtr intersection_program() const;
    Program::ConstPtr boundingbox_program()  const;

    optix::Geometry geometry()   const;
    operator optix::Geometry()   const;
    optix::Geometry operator->() const;
};

};

#endif //_DEF_DEFINE_OPTIX_HELPERS_GEOMETRY_H_
