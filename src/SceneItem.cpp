#include <optix_helpers/SceneItem.h>

namespace optix_helpers {

SceneItem::Ptr SceneItem::New(const Context::ConstPtr& context,
                              const Model::Ptr& model,
                              const std::string& acceleration)
{
    return Ptr(new SceneItem(context, model, acceleration));
}

SceneItem::SceneItem(const Context::ConstPtr& context,
                     const Model::Ptr& model,
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

void SceneItem::set_model(const Model::Ptr& model)
{
    if(model) {
        model_ = model;
        geomGroup_->setChild(0, *model_);
    }
}

void SceneItem::set_pose(const float* mat, const float* inv, bool transpose)
{
    transform_->setMatrix(transpose, mat, inv);
}

void SceneItem::set_pose(const Pose& pose)
{
    this->set_pose(pose.homogeneous_matrix().data(), NULL, true);
}

Model::Ptr SceneItem::model()
{
    return model_;
}

Model::ConstPtr SceneItem::model() const
{
    return model_;
}

SceneItem::Pose SceneItem::pose() const
{
    Matrix4 h;
    transform_->getMatrix(true, h.data(), NULL);
    return Pose::from_homogeneous_matrix(h);
}

optix::GeometryGroup SceneItem::geometry_group() const
{
    return geomGroup_;
}

optix::Transform SceneItem::transform() const
{
    return transform_;
}

optix::Transform SceneItem::node() const
{
    return this->transform();
}

}; //namespace optix_helpers
