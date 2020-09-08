#ifndef _DEF_DEFINE_OPTIX_HELPERS_GEOMETRY_H_
#define _DEF_DEFINE_OPTIX_HELPERS_GEOMETRY_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Program.h>

namespace optix_helpers {

class GeometryObj
{
    protected:
    
    optix::Geometry geometry_;
    Program intersectionProgram_;
    Program boundingboxProgram_;

    public:

    GeometryObj(const optix::Geometry& geometry,
                const Program& intersectionProgram,
                const Program& boundingboxProgram,
                size_t primitiveCount);
    
    Program intersection_program() const;
    Program boundingbox_program()  const;

    optix::Geometry geometry()   const;
    operator optix::Geometry()   const;
    optix::Geometry operator->() const;
};
using Geometry = std::shared_ptr<GeometryObj>;

};

#endif //_DEF_DEFINE_OPTIX_HELPERS_GEOMETRY_H_
