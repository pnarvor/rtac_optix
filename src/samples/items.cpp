#include <optix_helpers/samples/items.h>

namespace optix_helpers { namespace samples { namespace items {

SceneItem cube(const Context& context, const raytypes::RGB& rayType,
               float halfSize, const std::array<float,3> color)
{
    Model model = models::cube(context, halfSize);
    model->add_material(materials::rgb(context, rayType, color));
    return context->create_scene_item(model);
}

SceneItem sphere(const Context& context, const raytypes::RGB& rayType,
                 float radius, const std::array<float,3> color)
{
    Model model = models::sphere(context, radius);
    model->add_material(materials::rgb(context, rayType, color));
    return context->create_scene_item(model);
}

}; //namespace items
}; //namespace samples
}; //namespace optix_helpers


