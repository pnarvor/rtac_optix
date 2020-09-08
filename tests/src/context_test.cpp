#include <iostream>
using namespace std;

#include <optixu/optixpp.h>

#include <optix_helpers/Context.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    Context context;
    Source drawColor(cusample::drawColor, "draw_solid_color");
    Source alpha(cusample::alphaHeader, "alpha.h");
    
    Program program = context.create_program(drawColor, {alpha});
    cout << program << endl;

    context->setRayTypeCount(5);
    cout << "Num ray types : " << context->getRayTypeCount() << endl;

    return 0;
}


