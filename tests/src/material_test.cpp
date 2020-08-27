#include <iostream>
using namespace std;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/Material.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    Context context;

    RayType rayType0 = context->create_raytype(Source(cusample::coloredRay, "colored_ray.h"));
    cout << rayType0 << endl;

    Material white0(context);
    white0->add_closest_hit_program(rayType0, Source(cusample::whiteMaterial, "closest_hit_white"));

    cout << white0->get_closest_hit_program(rayType0) << endl;

    return 0;
}

