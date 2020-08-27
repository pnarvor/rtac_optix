#include <iostream>
#include <optix.h>
#include <exception>

#include <sutil.h>

#include <optix_helpers/Nvrtc.h>

#include "cusamples.h"

using namespace std;

int main(int argc, char** argv)
{
    RTcontext context = 0;

    RTprogram ray_gen_program;
    RTbuffer  buffer;

    /* Parameters */
    RTvariable result_buffer;
    RTvariable draw_color;

    int width  = 512u;
    int height = 384u;
    int i;

    /* Create our objects and set state */
    RT_CHECK_ERROR( rtContextCreate( &context ) );
    RT_CHECK_ERROR( rtContextSetRayTypeCount( context, 0 ) );
    RT_CHECK_ERROR( rtContextSetEntryPointCount( context, 1 ) );

    RT_CHECK_ERROR( rtBufferCreate( context, RT_BUFFER_OUTPUT, &buffer ) );
    RT_CHECK_ERROR( rtBufferSetFormat( buffer, RT_FORMAT_FLOAT4 ) );
    RT_CHECK_ERROR( rtBufferSetSize2D( buffer, width, height ) );
    RT_CHECK_ERROR( rtContextDeclareVariable( context, "result_buffer", &result_buffer ) );
    RT_CHECK_ERROR( rtVariableSetObject( result_buffer, buffer ) );
    
    optix_helpers::Nvrtc nvrtc;
    auto ptx = nvrtc.compile(cusample::drawColor, "draw_solid_color", {cusample::alphaHeader}, {"alpha.h"});
    RT_CHECK_ERROR( rtProgramCreateFromPTXString( context, ptx.c_str(), "draw_solid_color", &ray_gen_program) );
    RT_CHECK_ERROR( rtProgramDeclareVariable( ray_gen_program, "draw_color", &draw_color ) );
    RT_CHECK_ERROR( rtVariableSet3f( draw_color, 0.462f, 0.725f, 0.0f ) );
    RT_CHECK_ERROR( rtContextSetRayGenerationProgram( context, 0, ray_gen_program ) );

    /* Run */
    RT_CHECK_ERROR( rtContextValidate( context ) );
    RT_CHECK_ERROR( rtContextLaunch2D( context, 0 /* entry point */, width, height ) );
    
    sutil::initGlut(&argc, argv);
    sutil::displayBufferGlut( argv[0], buffer );
    RT_CHECK_ERROR( rtBufferDestroy( buffer ) );
    RT_CHECK_ERROR( rtProgramDestroy( ray_gen_program ) );
    RT_CHECK_ERROR( rtContextDestroy( context ) );

    cout << "ok" << endl;
    return 0;
}

