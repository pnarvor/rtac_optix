#include <iostream>
using namespace std;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/Material.h>
#include <optix_helpers/Geometry.h>
#include <optix_helpers/Model.h>
using namespace optix_helpers;

#include <optix_helpers/samples/raytypes.h>
using namespace optix_helpers::samples;

#include "cusamples.h"

int main()
{
    auto context = Context::New();

    raytypes::RGB rayType0(context);
    cout << rayType0 << endl;

    auto white = Material::New(context);
    white->add_closest_hit_program(rayType0,
        context->create_program(Source::New(cusample::whiteMaterial, "closest_hit_white"),
                                {rayType0.definition()}));
    cout << white->get_closest_hit_program(rayType0) << endl;

    auto sphere0 = Model::New(context);
    sphere0->set_geometry(Geometry::New(context,
        context->create_program(Source::New(cusample::sphere, "intersection")),
        context->create_program(Source::New(cusample::sphere, "bounds")),
        1));
    sphere0->add_material(white);

    return 0;
}

