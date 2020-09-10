#include <optix_helpers/samples/models.h>

#include <optix_helpers/samples/geometries.h>

namespace optix_helpers { namespace samples { namespace models {

Model cube(const Context& context, float halfSize)
{
    Model model = context->create_model();
    model->set_geometry(geometries::cube(context, halfSize));
    return model;
}

Model sphere(const Context& context, float radius)
{
    Model model = context->create_model();
    model->set_geometry(geometries::sphere(context, radius));
    return model;
}

}; //namespace geometries
}; //namespace samples
}; //namespace optix_helpers

