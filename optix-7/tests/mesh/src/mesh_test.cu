#include "mesh_test.h"


extern "C" {
    __constant__ Params params;
}

extern "C" __global__ void __raygen__mesh_test()
{
}

extern "C" __global__ void __miss__mesh_test()
{
}
