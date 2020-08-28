#include <optix_helpers/Model.h>

namespace optix_helpers
{

ModelObj::ModelObj(const optix::GeometryInstance& geomInstance,
                   const optix::Transform& pose) :
    geomInstance_(geomInstance),
    pose_(pose)
{}

optix::GeometryInstance ModelObj::geometry_instance() const
{
    return geomInstance_;
}

optix::Transform ModelObj::pose() const
{
    return pose_;
}

Model::Model() :
    Handle<ModelObj>()
{}

Model::Model(const optix::GeometryInstance& geomInstance,
             const optix::Transform& pose) :
    Handle<ModelObj>(new ModelObj(geomInstance, pose))
{}


};
