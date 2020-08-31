#ifndef _DEF_OPTIX_HELPERS_CONTEXT_H_
#define _DEF_OPTIX_HELPERS_CONTEXT_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Nvrtc.h>
#include <optix_helpers/Source.h>
#include <optix_helpers/Program.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/Material.h>
#include <optix_helpers/Geometry.h>
#include <optix_helpers/GeometryTriangles.h>
#include <optix_helpers/Model.h>

namespace optix_helpers {

class ContextObj
{
    protected:
    
    // Fix the mutable keyword use
    mutable optix::Context context_;
    mutable Nvrtc nvrtc_;

    public:

    ContextObj();

    optix::Context context() const; //? should be const ?
    unsigned int num_raytypes() const;

    Program create_program(const Source& source,
                           const Sources& additionalHeaders = Sources()) const; 

    RayType  create_raytype(const Source& rayDefinition) const;
    Material create_material() const;
    Geometry create_geometry(const Program& intersection = Program(),
                             const Program& boundingbox = Program()) const;
    GeometryTriangles create_geometry_triangles() const;
    Model create_model() const;
};

class Context : public Handle<ContextObj>
{
    public:

    Context();
};

}; // namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_CONTEXT_H_
