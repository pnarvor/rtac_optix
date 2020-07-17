#include <iostream>
using namespace std;

#include <optixu/optixpp.h>

#include <optix_helpers/ProgramManager.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    optix::Context context = optix::Context::create();
    ProgramManager manager(context);
    
    Program program = manager.from_custring(cusample::drawColor, "draw_solid_color");
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

