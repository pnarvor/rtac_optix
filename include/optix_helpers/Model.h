#ifndef _DEF_OPTIX_HELPERS_MODEL_H_
#define _DEF_OPTIX_HELPERS_MODEL_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/Geometry.h>
#include <optix_helpers/GeometryTriangles.h>
#include <optix_helpers/Material.h>

namespace optix_helpers
{

class ModelObj
{
    protected:

    optix::GeometryInstance geomInstance_;
    Geometry                geometry_;
    GeometryTriangles       geometryTriangles_;
    Materials               materials_;

    public:

    ModelObj(const Context& context);

    void set_geometry(const Geometry& geometry);
    void set_geometry(const GeometryTriangles& geometry);
    void add_material(const Material& material);

    optix::GeometryInstance geometry_instance() const;
    operator optix::GeometryInstance()          const;
    optix::GeometryInstance operator->()        const;
};

using Model = Handle<ModelObj>;


};

#endif //_DEF_OPTIX_HELPERS_SCENE_ITEMS_H_
