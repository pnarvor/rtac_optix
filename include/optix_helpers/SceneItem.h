#ifndef _DEF_OPTIX_HELPERS_SCENE_ITEM_H_
#define _DEF_OPTIX_HELPERS_SCENE_ITEM_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Model.h>

namespace optix_helpers {

class SceneItemObj
{
    protected:

    Model model_;
    optix::GeometryGroup geomGroup_;
    optix::Transform     transform_;

    public:

    SceneItemObj(const optix::GeometryGroup& geomGroup,
                 const optix::Transform& transform,
                 const optix::Acceleration& acceleration = optix::Acceleration(),
                 const Model& model = Model());
    
    void set_pose(const float* mat, const float* inv = NULL, 
                  bool transpose = false);
    void set_acceleration(const optix::Acceleration& acceleration);
    void set_model(const Model& model);

    Model model() const;
    optix::GeometryGroup geometry_group() const;
    optix::Transform     transform() const;
    optix::Transform     node() const;
};

class SceneItem : public Handle<SceneItemObj>
{
    public:

    SceneItem();
    SceneItem(const optix::GeometryGroup& geomGroup,
              const optix::Transform& transform,
              const optix::Acceleration& acceleration = optix::Acceleration(),
              const Model& model = Model());
};

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SCENE_ITEM_H_
