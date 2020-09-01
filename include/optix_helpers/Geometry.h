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
                const Program& intersectionProgram = Program(),
                const Program& boundingboxProgram = Program(),
                size_t primitiveCount = 1);
    
    void set_intersection_program(const Program& program);
    void set_boundingbox_program(const Program& program);
    void set_primitive_count(size_t primitiveCount);

    optix::Geometry geometry()     const;
    Program intersection_program() const;
    Program boundingbox_program()  const;
    size_t primitive_count()       const;
};

class Geometry : public Handle<GeometryObj>
{
    public:

    Geometry();
    Geometry(const optix::Geometry& geometry,
             const Program& intersectionProgram = Program(),
             const Program& boundingboxProgram = Program(),
             size_t primitiveCount = 1);

    operator optix::Geometry() const;
};

};

#endif //_DEF_DEFINE_OPTIX_HELPERS_GEOMETRY_H_
