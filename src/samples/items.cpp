#include <optix_helpers/samples/items.h>

namespace optix_helpers { namespace samples { namespace items {

SceneItem cube(const Context& context, const Materials& materials, float scale)
{
    Model model = context->create_model();
    model->set_geometry(geometries::cube(context, scale));
    for(auto material : materials) {
        model->add_material(material);
    }
    return context->create_scene_item(model);
}

SceneItem sphere(const Context& context, const Materials& materials, float radius)
{
    Model model = context->create_model();
    model->set_geometry(geometries::sphere(context, radius));
    for(auto material : materials) {
        model->add_material(material);
    }
    return context->create_scene_item(model);
}

SceneItem square(const Context& context, const Materials& materials, float scale)
{
    Model model = context->create_model();
    model->set_geometry(geometries::square(context, scale));
    for(auto material : materials) {
        model->add_material(material);
    }
    return context->create_scene_item(model);
}

SceneItem tube(const Context& context, const Materials& materials,
               float radius, float height)
{
    Model model = context->create_model();
    model->set_geometry(geometries::tube(context, radius, height));
    for(auto material : materials) {
        model->add_material(material);
    }
    return context->create_scene_item(model);
}

SceneItem parabola(const Context& context, const Materials& materials,
                   float a, float b, float height)
{
    Model model = context->create_model();
    model->set_geometry(geometries::parabola(context, a, b, height));
    for(auto material : materials) {
        model->add_material(material);
    }
    return context->create_scene_item(model);
}

}; //namespace items
}; //namespace samples
}; //namespace optix_helpers


