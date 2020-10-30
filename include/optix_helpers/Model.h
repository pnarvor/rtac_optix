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

class Model
{
    public:

    using Ptr      = Handle<Model>;
    using ConstPtr = Handle<const Model>;

    protected:

    optix::GeometryInstance geomInstance_;
    Geometry::Ptr           geometry_;
    GeometryTriangles::Ptr  geometryTriangles_;
    Materials               materials_;

    public:

    static Ptr New(const Context::ConstPtr& context);
    Model(const Context::ConstPtr& context);

    void set_geometry(const Geometry::Ptr& geometry);
    void set_geometry(const GeometryTriangles::Ptr& geometry);
    void add_material(const Material::Ptr& material);

    optix::GeometryInstance geometry_instance() const;
    operator optix::GeometryInstance()          const;
    optix::GeometryInstance operator->()        const;
};

};

#endif //_DEF_OPTIX_HELPERS_SCENE_ITEMS_H_
