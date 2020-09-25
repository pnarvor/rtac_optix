#include <optix_helpers/Scene.h>

namespace optix_helpers {

SceneObj::SceneObj(const std::string& topObjectName, bool loadDefaultConfig) :
    ContextObj(),
    topObject_(context_->createGroup())
{
    topObject_->setAcceleration(context_->createAcceleration("Trbvh"));
    context_[topObjectName]->set(topObject_);

    if(loadDefaultConfig)
        this->load_default_optix_config();
}

void SceneObj::load_default_optix_config()
{
    using namespace std;
    cout << "Default trace depth : " << context_->getMaxTraceDepth() << endl;
    cout << "Default max callable program : " << context_->getMaxCallableProgramDepth() << endl;
    cout << "Default stack size : " << context_->getStackSize() << endl;

    context_->setMaxTraceDepth(10);
    context_->setMaxCallableProgramDepth(10);
    context_->setStackSize(8096);

    using namespace std;
    cout << "Trace depth : " << context_->getMaxTraceDepth() << endl;
    cout << "Max callable program : " << context_->getMaxCallableProgramDepth() << endl;
    cout << "Stack size : " << context_->getStackSize() << endl;
}

void SceneObj::add_child(const SceneItem& item)
{
    topObject_->addChild(item->node());
}

Scene::Scene(const std::string& topObjectName, bool loadDefaultConfig) :
    Handle<SceneObj>(topObjectName, loadDefaultConfig)
{}

Scene::operator Context()
{
    return Context(std::dynamic_pointer_cast<ContextObj>(this->obj_));
}

}; //namespace optix_helpers

