#include <iostream>
using namespace std;

#include <optixu/optixpp.h>

#include <optix_helpers/Context.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    auto context = Context::New();
    auto drawColor = Source::New(cusample::drawColor, "draw_solid_color");
    auto alpha     = Source::New(cusample::alphaHeader, "alpha.h");
    
    auto program = context->create_program(drawColor, {alpha});
    cout << program << endl;

    (*context)->setRayTypeCount(5);
    cout << "Num ray types : " << (*context)->getRayTypeCount() << endl;

    return 0;
}


