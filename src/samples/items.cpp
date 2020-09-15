#include <optix_helpers/samples/items.h>

namespace optix_helpers { namespace samples { namespace items {

SceneItem cube(const Context& context, const Materials& materials, float scale)
{
    Model model = models::cube(context, scale);
    for(auto material : materials) {
        model->add_material(material);
    }
    return context->create_scene_item(model);
}

SceneItem sphere(const Context& context, const Materials& materials, float radius)
{
    Model model = models::sphere(context, radius);
    for(auto material : materials) {
        model->add_material(material);
    }
    return context->create_scene_item(model);
}

SceneItem square(const Context& context, const Materials& materials, float scale)
{
    Model model = models::square(context, scale);
    for(auto material : materials) {
        model->add_material(material);
    }
    return context->create_scene_item(model);
}

}; //namespace items
}; //namespace samples
}; //namespace optix_helpers


