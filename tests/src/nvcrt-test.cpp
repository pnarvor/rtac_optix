#include <iostream>
#include <optix_helpers/NVRTC_Helper.h>

#include "cusamples.h"

using namespace std;
using namespace optix_helpers;

int main()
{
    NVRTC_Helper nvrtc;
    //cout << nvrtc << endl;
    
    auto ptx = nvrtc.compile(cusample::drawColor, "draw_color",
                             {cusample::alphaHeader}, {"alpha.h"});
    cout << ptx << endl;

    Source drawColor(cusample::drawColor, "draw_color");
    Source alpha(cusample::alphaHeader, "alpha.h");
    auto ptx1 = nvrtc.compile(drawColor, {alpha});
    cout << ptx1 << endl;
    return 0;
}

