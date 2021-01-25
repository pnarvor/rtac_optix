#include <iostream>
using namespace std;

#include <optix_helpers/Context.h>
#include <optix_helpers/Buffer.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    auto context = Context::New();
    
    auto deviceCount =  context->context()->getDeviceCount();
    cout << "Device count : " << deviceCount << endl;

    auto buffer = Buffer::New(context, RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, "buffer");
    buffer->set_size(1920, 1080);

    return 0;
}

