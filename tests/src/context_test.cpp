#include <iostream>
using namespace std;

#include <optixu/optixpp.h>

#include <optix_helpers/Context.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    Context context = create_context();
    Source drawColor = create_source(cusample::drawColor, "draw_solid_color");
    Source alpha = create_source(cusample::alphaHeader, "alpha.h");
    
    Program program = context->create_program(drawColor, {alpha});
    cout << program << endl;

    (*context)->setRayTypeCount(5);
    cout << "Num ray types : " << (*context)->getRayTypeCount() << endl;

    return 0;
}


