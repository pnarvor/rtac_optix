#include <optix_helpers/samples/models.h>

#include <optix_helpers/samples/geometries.h>

namespace optix_helpers { namespace samples { namespace models {

Model cube(const Context& context, float scale)
{
    Model model = context->create_model();
    model->set_geometry(geometries::cube(context, scale));
    return model;
}

Model sphere(const Context& context, float radius)
{
    Model model = context->create_model();
    model->set_geometry(geometries::sphere(context, radius));
    return model;
}

Model cube_with_attributes(const Context& context, float scale)
{
    Model model = context->create_model();
    model->set_geometry(geometries::cube_with_attributes(context, scale));
    return model;
}

}; //namespace geometries
}; //namespace samples
}; //namespace optix_helpers

