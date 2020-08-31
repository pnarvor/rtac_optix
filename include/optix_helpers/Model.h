#ifndef _DEF_OPTIX_HELPERS_MODEL_H_
#define _DEF_OPTIX_HELPERS_MODEL_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Geometry.h>
#include <optix_helpers/GeometryTriangles.h>

namespace optix_helpers
{

class ModelObj
{
    protected:

    optix::GeometryInstance geomInstance_;
    optix::Transform        pose_;
    Geometry                geometry_;
    GeometryTriangles       geometryTriangles_;

    public:

    ModelObj(const optix::GeometryInstance& geomInstance,
             const optix::Transform& pose);

    void set_geometry(const Geometry& geometry);
    void set_geometry(const GeometryTriangles& geometry);
    void set_pose(const float* mat, bool transpose = false,
                  const float* inverted = NULL);

    optix::GeometryInstance geometry_instance() const;
    optix::Transform pose() const;
    optix::Transform node() const;
};

class Model : public Handle<ModelObj>
{
    public:

    Model();
    Model(const optix::GeometryInstance& geomInstance,
          const optix::Transform& pose);
};

};

#endif //_DEF_OPTIX_HELPERS_SCENE_ITEMS_H_
