#include <iostream>
using namespace std;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/Material.h>
#include <optix_helpers/Model.h>
#include <optix_helpers/RayGenerator.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    Context context;

    RayType rayType0 = context->create_raytype(Source(cusample::coloredRay, "colored_ray.h"));
    //cout << rayType0 << endl;

    Material white(context->create_material());
    white->add_closest_hit_program(rayType0,
        context->create_program(Source(cusample::whiteMaterial, "closest_hit_white"),
                                {rayType0->definition()}));
    //cout << white->get_closest_hit_program(rayType0) << endl;

    Model sphere0 = context->create_model();
    sphere0->set_geometry(context->create_geometry(
        context->create_program(Source(cusample::sphere, "intersection")),
        context->create_program(Source(cusample::sphere, "bounds"))));
    sphere0->add_material(white);

    auto renderer = context->create_raygenerator(32,16,1);
    renderer->set_raygen_program(context->create_raygen_program(
        "render_buffer", Source(cusample::rayGenOrtho, "ortho_z"), {rayType0->definition()}));
    renderer->set_miss_program(context->create_program(
        Source(cusample::coloredMiss, "black_miss"), {rayType0->definition()}));

    return 0;
}

