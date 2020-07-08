#include <iostream>
#include <optix_helpers/NVRTC_Helper.h>

#include "cusamples.h"

using namespace std;
using namespace optix;

int main()
{
    NVRTC_Helper nvrtc;
    //cout << nvrtc << endl;
    
    auto ptx = nvrtc.compile(cusample::drawColor);
    cout << ptx << endl;

    return 0;
}

