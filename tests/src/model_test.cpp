#include <iostream>
using namespace std;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/Material.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    Context context = create_context();
    
    RayType rayType0 = context->create_raytype(create_source(cusample::coloredRay, "colored_ray.h"));
    cout << rayType0 << endl;

    Material white(context->create_material());
    white->add_closest_hit_program(rayType0,
        context->create_program(create_source(cusample::whiteMaterial, "closest_hit_white"),
                                {rayType0->definition()}));
    cout << white->get_closest_hit_program(rayType0) << endl;

    Model sphere0 = context->create_model();
    sphere0->set_geometry(context->create_geometry(
        context->create_program(create_source(cusample::sphere, "intersection")),
        context->create_program(create_source(cusample::sphere, "bounds")),
        1));
    sphere0->add_material(white);

    return 0;
}

