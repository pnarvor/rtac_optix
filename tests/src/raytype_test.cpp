#include <iostream>
using namespace std;

#include <optix_helpers/RayType.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    RayType rayType0(0, Source(cusample::coloredRay, "colored_ray.h"));
    cout << rayType0 << endl;

    return 0;
}

