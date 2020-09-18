#ifndef _DEF_OPTIX_HELPERS_SCENE_ITEM_H_
#define _DEF_OPTIX_HELPERS_SCENE_ITEM_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <rtac_base/types/Pose.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Model.h>

namespace optix_helpers {

class SceneItemObj
{
    public:

    using Pose    = rtac::types::Pose<float>;
    using Matrix4 = rtac::types::Matrix4<float>;

    protected:

    Model model_;
    optix::GeometryGroup geomGroup_;
    optix::Transform     transform_;

    public:

    SceneItemObj(const optix::GeometryGroup& geomGroup,
                 const optix::Transform& transform,
                 const optix::Acceleration& acceleration,
                 const Model& model);
    
    void set_pose(const float* mat, const float* inv = NULL, 
                  bool transpose = false);
    void set_pose(const Pose& pose);
    void set_acceleration(const optix::Acceleration& acceleration);
    void set_model(const Model& model);

    Model model() const;
    optix::GeometryGroup geometry_group() const;
    optix::Transform     transform() const;
    optix::Transform     node() const;
    Pose                 pose() const;
};
using SceneItem = Handle<SceneItemObj>;

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SCENE_ITEM_H_
