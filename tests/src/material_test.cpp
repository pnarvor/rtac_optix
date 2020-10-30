#include <iostream>
using namespace std;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/Material.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    auto context = Context::New();

    RayType rayType0 = context->instanciate_raytype(Source::New(cusample::coloredRay, "colored_ray.h"));
    cout << rayType0 << endl;

    auto white0 = Material::New(context);
    white0->add_closest_hit_program(rayType0,
        context->create_program(Source::New(cusample::whiteMaterial, "closest_hit_white"),
                                {rayType0.definition()}));

    cout << white0->get_closest_hit_program(rayType0) << endl;
    (*white0)->getContext()->setRayTypeCount(10);

    return 0;
}

