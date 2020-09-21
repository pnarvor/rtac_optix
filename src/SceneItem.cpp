#include <optix_helpers/SceneItem.h>

namespace optix_helpers {

SceneItemObj::SceneItemObj(const Context& context,
                           const Model& model,
                           const std::string& acceleration) :
    model_(model),
    geomGroup_((*context)->createGeometryGroup()),
    transform_((*context)->createTransform())
{
    transform_->setChild(geomGroup_);
    geomGroup_->setAcceleration((*context)->createAcceleration(acceleration.c_str()));
    if(model_)
        geomGroup_->addChild(*model_);
}

void SceneItemObj::set_pose(const float* mat, const float* inv, bool transpose)
{
    transform_->setMatrix(transpose, mat, inv);
}

void SceneItemObj::set_pose(const Pose& pose)
{
    this->set_pose(pose.homogeneous_matrix().data(), NULL, true);
}

void SceneItemObj::set_model(const Model& model)
{
    if(model) {
        model_ = model;
        geomGroup_->setChild(0, *model_);
    }
}

Model SceneItemObj::model() const
{
    return model_;
}

optix::GeometryGroup SceneItemObj::geometry_group() const
{
    return geomGroup_;
}

optix::Transform SceneItemObj::transform() const
{
    return transform_;
}

optix::Transform SceneItemObj::node() const
{
    return this->transform();
}

SceneItemObj::Pose SceneItemObj::pose() const
{
    Matrix4 h;
    transform_->getMatrix(true, h.data(), NULL);
    return Pose::from_homogeneous_matrix(h);
}

}; //namespace optix_helpers
