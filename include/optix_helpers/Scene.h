#ifndef _DEF_OPTIX_HELPERS_SCENE_H_
#define _DEF_OPTIX_HELPERS_SCENE_H_

#include <iostream>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/SceneItem.h>

namespace optix_helpers {

class SceneObj : public ContextObj
{
    protected:

    optix::Group topObject_;

    void load_default_optix_config();

    public:

    SceneObj(const std::string& topObjectName = "topObject",
             bool loadDefaultConfig = true);

    void add_child(const SceneItem& item);
};
class Scene : public Handle<SceneObj>
{
    public:

    Scene(const std::string& topObjectName = "topObject",
          bool loadDefaultConfig = true);

    operator Context();
};

}; //namespace optix_helpers
#endif //_DEF_OPTIX_HELPERS_SCENE_H_

