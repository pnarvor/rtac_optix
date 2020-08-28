#ifndef _DEF_OPTIX_HELPERS_MODEL_H_
#define _DEF_OPTIX_HELPERS_MODEL_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>

namespace optix_helpers
{

class ModelObj
{
    protected:

    optix::GeometryInstance geomInstance_;
    optix::Transform        pose_;

    public:

    ModelObj(const optix::GeometryInstance& geomInstance,
             const optix::Transform& pose);

    optix::GeometryInstance geometry_instance() const;
    optix::Transform pose() const;
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
