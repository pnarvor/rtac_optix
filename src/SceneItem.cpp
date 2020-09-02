#include <optix_helpers/SceneItem.h>

namespace optix_helpers {

SceneItemObj::SceneItemObj(const optix::GeometryGroup& geomGroup,
             const optix::Transform& transform,
             const Model& model) :
    model_(model),
    geomGroup_(geomGroup),
    transform_(transform)
{
    transform_->setChild(geomGroup_);
    if(model_)
        geomGroup_->addChild(model_);
}

void SceneItemObj::set_pose(const float* mat, const float* inv, bool transpose)
{
    transform_->setMatrix(transpose, mat, inv);
}

void SceneItemObj::set_acceleration(const optix::Acceleration& acceleration)
{
    geomGroup_->setAcceleration(acceleration);
}

void SceneItemObj::set_model(const Model& model)
{
    if(model) {
        model_ = model;
        geomGroup_->setChild(0, model_);
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

SceneItem::SceneItem() :
    Handle<SceneItemObj>()
{}

SceneItem::SceneItem(const optix::GeometryGroup& geomGroup,
                     const optix::Transform& transform,
                     const Model& model) :
    Handle<SceneItemObj>(new SceneItemObj(geomGroup, transform, model))
{}

}; //namespace optix_helpers
