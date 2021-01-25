#ifndef _DEF_OPTIX_HELPERS_SCENE_H_
#define _DEF_OPTIX_HELPERS_SCENE_H_

#include <iostream>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/SceneItem.h>

namespace optix_helpers {

class Scene : public Context
{
    public:

    using Ptr      = Handle<Scene>;
    using ConstPtr = Handle<const Scene>;

    protected:

    optix::Group topObject_;

    void load_default_optix_config();

    public:

    static Ptr New(const std::string& topObjectName = "topObject",
                   bool loadDefaultConfig = true);
    Scene(const std::string& topObjectName = "topObject",
          bool loadDefaultConfig = true);

    void add_child(const SceneItem::Ptr& item);
};

}; //namespace optix_helpers
#endif //_DEF_OPTIX_HELPERS_SCENE_H_

