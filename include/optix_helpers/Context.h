#ifndef _DEF_OPTIX_HELPERS_CONTEXT_H_
#define _DEF_OPTIX_HELPERS_CONTEXT_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Nvrtc.h>
#include <optix_helpers/Source.h>
#include <optix_helpers/Program.h>
#include <optix_helpers/Buffer.h>
//#include <optix_helpers/RayType.h>
//#include <optix_helpers/Material.h>
//#include <optix_helpers/Geometry.h>
//#include <optix_helpers/GeometryTriangles.h>
//#include <optix_helpers/Model.h>
//#include <optix_helpers/SceneItem.h>
//#include <optix_helpers/RayGenerator.h>

namespace optix_helpers {

class Context
{
    protected:
    
    // Fix the mutable keyword use
    mutable optix::Context context_;
    mutable Nvrtc nvrtc_;

    public:

    Context();

    Program create_program(const Source& source,
                           const Sources& additionalHeaders = Sources()) const; 
    Buffer create_buffer(RTbuffertype bufferType, const std::string& name = "buffer") const;

    //RayType  create_raytype(const Source& rayDefinition) const;
    //Material create_material() const;
    //Geometry create_geometry(const Program& intersection = Program(),
    //                         const Program& boundingbox = Program(),
    //                         size_t primitiveCount = 1) const;
    //GeometryTriangles create_geometry_triangles() const;
    //template <typename Tp, typename Tf>
    //GeometryTriangles create_mesh(size_t numPoints, const Tp* points,
    //                              size_t numFaces,  const Tf* faces) const;
    //Model create_model() const;
    //RayGenerator create_raygenerator(size_t width, size_t height, size_t depth=1) const;
    //SceneItem create_scene_item(const Model& model, const char* acceleration = "Trbvh") const;


    optix::Handle<optix::VariableObj> operator[](const std::string& varname);
    operator optix::Context()   const;
    optix::Context operator->();
    optix::Context operator->() const;
    optix::Context context()    const; //? should be const ?
};

//template <typename Tp, typename Tf>
//GeometryTriangles ContextObj::create_mesh(size_t numPoints, const Tp* points,
//                                          size_t numFaces,  const Tf* faces) const
//{
//    GeometryTriangles mesh = this->create_geometry_triangles();
//    mesh->set_points(numPoints, points);
//    mesh->set_faces(numFaces, faces);
//    return mesh;
//}

}; // namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_CONTEXT_H_
