#include <iostream>
using namespace std;

#include <optix_helpers/Source.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    Source src0 = Source(cusample::alphaHeader, "alpha.h");
    cout << src0 << endl;
    return 0;
}


