#include <iostream>
#include <optix_helpers/Nvrtc.h>
#include <optix_helpers/Source.h>

#include "cusamples.h"

using namespace std;
using namespace optix_helpers;

int main()
{
    NVRTC_Helper nvrtc;

    Source drawColor = Source(cusample::drawColor, "draw_color");
    Source alpha = Source(cusample::alphaHeader, "alpha.h");
    auto ptx1 = nvrtc.compile(drawColor, {alpha});
    cout << ptx1 << endl;
    return 0;
}

