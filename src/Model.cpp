#include <optix_helpers/Model.h>

namespace optix_helpers
{

ModelObj::ModelObj(const optix::GeometryInstance& geomInstance,
                   const optix::Transform& pose) :
    geomInstance_(geomInstance),
    pose_(pose)
{
    pose_->setChild(geomInstance_);
}

void ModelObj::set_geometry(const Geometry& geometry)
{
    geometry_          = geometry;
    geometryTriangles_ = GeometryTriangles();
    geomInstance_->setGeometry(geometry_);
}

void ModelObj::set_geometry(const GeometryTriangles& geometry)
{
    geometry_          = Geometry();
    geometryTriangles_ = geometry;
    geomInstance_->setGeometryTriangles(geometryTriangles_);
}

void ModelObj::set_pose(const float* mat, bool transpose, const float* inverted)
{
    //expects a 4x4 row-major homogeneous matrix.
    pose_->setMatrix(transpose, mat, inverted);
}

optix::GeometryInstance ModelObj::geometry_instance() const
{
    return geomInstance_;
}

optix::Transform ModelObj::pose() const
{
    return pose_;
}

optix::Transform ModelObj::node() const
{
    return this->pose();
}

Model::Model() :
    Handle<ModelObj>()
{}

Model::Model(const optix::GeometryInstance& geomInstance,
             const optix::Transform& pose) :
    Handle<ModelObj>(new ModelObj(geomInstance, pose))
{}


};
