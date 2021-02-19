#include <iostream>
using namespace std;

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#include <rtac_base/cuda/utils.h>
#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
using namespace rtac::cuda;

#include <rtac_optix/utils.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/Pipeline.h>
using namespace rtac::optix;

#include <rtac_optix/samples/PinholeCamera.h>
using namespace rtac::optix::samples;

#include <pinhole_test/ptx_files.h>

#include "pinhole_test.h"

using RaygenRecord = SbtRecord<RaygenData>;
using MissRecord   = SbtRecord<MissData>;

int main()
{
    unsigned int W = 16, H = 9;

    auto ptxFiles = pinhole_test::get_ptx_files();

    cudaFree(0);
    optixInit();

    auto context = Context::Create();
    auto pipeline = Pipeline::Create(context);
    pipeline->add_module("src/pinhole_test.cu", ptxFiles["src/pinhole_test.cu"]);
    auto raygenProgram = pipeline->add_raygen_program("__raygen__pinhole",
                                                     "src/pinhole_test.cu");
    // the missProgram is mandatory even without rays.
    auto missProgram = pipeline->add_miss_program("__miss__pinhole",
                                                 "src/pinhole_test.cu");
    pipeline->link();

    // Shader binding table setup = setting program parameters
    auto sbt = rtac::optix::zero<OptixShaderBindingTable>();
    // Is setting the record mandatory when empty ?
    RaygenRecord raygenRecord; // no parameters
    OPTIX_CHECK(optixSbtRecordPackHeader(*raygenProgram, &raygenRecord));
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(memcpy::host_to_device(raygenRecord));
    MissRecord missRecord; // no parameters
    OPTIX_CHECK(optixSbtRecordPackHeader(*missProgram, &missRecord));
    sbt.missRecordBase          = reinterpret_cast<CUdeviceptr>(memcpy::host_to_device(missRecord));
    sbt.missRecordCount         = 1;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);

    DeviceVector<float> imgData(W*H);

    Params params;
    params.width     = W;
    params.height    = H;
    params.imageData = imgData.data(); 
    params.cam = PinholeCamera::New();
    cout << params.cam << endl;

    CUstream stream;
    cudaStreamCreate(&stream);

    OPTIX_CHECK(optixLaunch(*pipeline, stream,
                            reinterpret_cast<CUdeviceptr>(memcpy::host_to_device(params)),
                            sizeof(params),
                            &sbt, W,H,1));
    cudaDeviceSynchronize();

    HostVector<float> output(imgData);
    for(int h = 0; h < H; h++) {
        for(int w = 0; w < W; w++) {
            cout << " " << output[h*W + w];
        }
        cout << endl;
    }

    return 0;
}

