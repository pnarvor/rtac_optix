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
#include <optix_helpers/Material.h>
#include <optix_helpers/Geometry.h>
#include <optix_helpers/GeometryTriangles.h>
#include <optix_helpers/Model.h>
#include <optix_helpers/SceneItem.h>
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

    RayType  create_raytype(const Source& rayDefinition) const;
    Material create_material() const;
    Geometry create_geometry(const Program& intersection,
                             const Program& boundingbox,
                             size_t primitiveCount) const;
    GeometryTriangles create_geometry_triangles() const;
    template <typename Tp, typename Tf>
    GeometryTriangles create_geometry_triangles(const rtac::types::Mesh<Tp,Tf,3>& mesh) const;
    Model create_model() const;
    SceneItem create_scene_item(const Model& model, const char* acceleration = "Trbvh") const;
    //RayGenerator create_raygenerator(size_t width, size_t height, size_t depth=1) const;


    optix::Handle<optix::VariableObj> operator[](const std::string& varname);
    operator optix::Context()   const;
    optix::Context operator->();
    optix::Context operator->() const;
    optix::Context context()    const; //? should be const ?
};

class Context : public Handle<ContextObj>
{
    public:

    Context(int entryPointCount = 1);
};

template <typename Tp, typename Tf>
GeometryTriangles ContextObj::create_geometry_triangles(const rtac::types::Mesh<Tp,Tf,3>& mesh) const
{
    GeometryTriangles geom(this->create_geometry_triangles());
    geom->set_mesh(mesh);
    return geom;
}

}; // namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_CONTEXT_H_
