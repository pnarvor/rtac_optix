#include <iostream>
using namespace std;

#include <optix_helpers/Context.h>
#include <optix_helpers/Buffer.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    Context context(create_context());

    Buffer buffer(context->create_buffer(RT_BUFFER_OUTPUT));

    (*buffer)->setFormat(RT_FORMAT_FLOAT);
    cout << "buffer format : " << (*buffer)->getFormat() << endl;
    (*buffer)->setFormat(RT_FORMAT_FLOAT2);
    cout << "buffer format : " << (*buffer)->getFormat() << endl;
    (*buffer)->setFormat(RT_FORMAT_FLOAT3);
    cout << "buffer format : " << (*buffer)->getFormat() << endl;
    (*buffer)->setFormat(RT_FORMAT_FLOAT4);
    cout << "buffer format : " << (*buffer)->getFormat() << endl;

    (*buffer)->setFormat(RT_FORMAT_BYTE);
    cout << "buffer format : " << (*buffer)->getFormat() << endl;
    (*buffer)->setFormat(RT_FORMAT_BYTE2);
    cout << "buffer format : " << (*buffer)->getFormat() << endl;
    (*buffer)->setFormat(RT_FORMAT_BYTE3);
    cout << "buffer format : " << (*buffer)->getFormat() << endl;
    (*buffer)->setFormat(RT_FORMAT_BYTE4);
    cout << "buffer format : " << (*buffer)->getFormat() << endl;

    getchar();

    return 0;
}

