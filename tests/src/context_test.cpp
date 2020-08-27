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
    
    Program program = context->create_program(drawColor, {alpha});
    cout << program << endl;

    Program nullProgram;
    if(!nullProgram) cout << "Program is null" << endl;
    //// Not compiling
    //if(nullProgram) {
    //    cout << "Program is not null" << endl;
    //}
    //else {
    //    cout << "Program is null" << endl;
    //}

    return 0;
}


