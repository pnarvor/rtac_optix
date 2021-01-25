#ifndef _DEF_OPTIX_HELPERS_SCENE_ITEM_H_
#define _DEF_OPTIX_HELPERS_SCENE_ITEM_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <rtac_base/types/Pose.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/Model.h>

namespace optix_helpers {

class SceneItem
{
    public:

    using Ptr      = Handle<SceneItem>;
    using ConstPtr = Handle<const SceneItem>;

    using Pose    = rtac::types::Pose<float>;
    using Matrix4 = rtac::types::Matrix4<float>;

    protected:

    Model::Ptr           model_;
    optix::GeometryGroup geomGroup_;
    optix::Transform     transform_;

    public:

    static Ptr New(const Context::ConstPtr& context,
                   const Model::Ptr& model,
                   const std::string& acceleration = "Trvbh");
    SceneItem(const Context::ConstPtr& context,
              const Model::Ptr& model,
              const std::string& acceleration = "Trvbh");
    
    void set_model(const Model::Ptr& model);
    void set_pose(const float* mat, const float* inv = NULL, 
                  bool transpose = false);
    void set_pose(const Pose& pose);

    Model::Ptr       model();

    Model::ConstPtr  model() const;
    Pose             pose()  const;

    optix::GeometryGroup geometry_group() const;
    optix::Transform     transform() const;
    optix::Transform     node() const;
};

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SCENE_ITEM_H_
