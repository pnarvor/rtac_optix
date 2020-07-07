#include <iostream>
#include <optix.h>
#include <exception>

#include "cusamples.h"

using namespace std;

// Error check/report helper for users of the C API                 
#define RT_CHECK_ERROR( func )                                     \
  do {                                                             \
    RTresult code = func;                                          \
    if( code != RT_SUCCESS )                                       \
      throw std::runtime_error(std::to_string(code)+__FILE__+":"+std::to_string(__LINE__));\
  } while(0)                                                        


int main()
{
	cout << "NVRTC_INCLUDE_DIRS : " << std::string(NVRTC_INCLUDE_DIRS) << endl;
    cout << "Hello There !" << endl;
	cout << cusample::drawColor << endl;

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

    return 0;
}

