#include <iostream>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac::files;

#include <rtac_base/cuda/utils.h>
#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/Texture2D.h>
using namespace rtac::cuda;

#include <rtac_optix/utils.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/Pipeline.h>
#include <rtac_optix/MeshAccelStruct.h>
using namespace rtac::optix;

#include <rtac_optix_7_sbt_test0/ptx_files.h>

#include "sbt_test0.h"

using RaygenRecord     = SbtRecord<RaygenData>;
using MissRecord       = SbtRecord<MissData>;
using ClosestHitRecord = SbtRecord<ClosestHitData>;

int main()
{
    unsigned int W = 800, H = 600;
    auto ptxFiles = rtac_optix_7_sbt_test0::get_ptx_files();

    rtac::cuda::init_cuda();
    OPTIX_CHECK( optixInit() );

    auto context  = Context::Create();

    // Building pipeline.
    auto pipeline = Pipeline::Create(context);
    pipeline->add_module("sbt_test0", ptxFiles["src/sbt_test0.cu"]);
    auto raygenProgram = pipeline->add_raygen_program("__raygen__sbt_test", "sbt_test0");
    auto missProgram   = pipeline->add_miss_program("__miss__sbt_test", "sbt_test0");
    auto hitDesc = rtac::optix::zero<OptixProgramGroupDesc>();
    hitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitDesc.hitgroup.moduleCH = pipeline->module("sbt_test0");
    hitDesc.hitgroup.entryFunctionNameCH = "__closesthit__sbt_test";
    auto hitProgram = pipeline->add_program_group(hitDesc);
    // At this point the pipeline is not linked and the program are not
    // compiled yet. They will do so when used in an optix API call. (the
    // implicit cast between rtac::optix type and corresponding OptiX native
    // types will trigger compilation / link.
    
    // Simple cube as scene
    auto topObject = MeshAccelStruct::Create(context, MeshAccelStruct::cube_data());
    topObject->add_sbt_flags(OPTIX_GEOMETRY_FLAG_NONE);

    auto checkerboardTex = Texture2D<uchar4>::checkerboard(4,4,
                                                           uchar4({255,255,0,255}),
                                                           uchar4({0,0,255,255}));
    
    // setting up sbt
    auto sbt = rtac::optix::zero<OptixShaderBindingTable>();

    RaygenRecord raygenRecord;
    OPTIX_CHECK( optixSbtRecordPackHeader(*raygenProgram, &raygenRecord) );
    sbt.raygenRecord = reinterpret_cast<CUdeviceptr>(
        rtac::cuda::memcpy::host_to_device(raygenRecord));

    MissRecord missRecord;
    OPTIX_CHECK( optixSbtRecordPackHeader(*missProgram, &missRecord) );
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(
        rtac::cuda::memcpy::host_to_device(missRecord));
    sbt.missRecordCount = 1;
    sbt.missRecordStrideInBytes = sizeof(MissRecord);

    ClosestHitRecord hitRecord;
    hitRecord.data.texObject = checkerboardTex;
    OPTIX_CHECK( optixSbtRecordPackHeader(*hitProgram, &hitRecord) );
    sbt.hitgroupRecordBase = reinterpret_cast<CUdeviceptr>(
        rtac::cuda::memcpy::host_to_device(hitRecord));
    sbt.hitgroupRecordCount = 1;
    sbt.hitgroupRecordStrideInBytes = sizeof(ClosestHitRecord);

    DeviceVector<uchar3> imgData(W*H);

    auto params = rtac::optix::zero<Params>();
    params.width     = W;
    params.height    = H;
    params.imageData = imgData.data();
    params.cam       = samples::PinholeCamera::New({0.0f,0.0f,0.0f}, {5.0f,4.0f,3.0f});
    params.sceneTreeHandle = *topObject;

    OPTIX_CHECK( optixLaunch(*pipeline, 0, 
                             reinterpret_cast<CUdeviceptr>(memcpy::host_to_device(params)),
                             sizeof(params), &sbt, W, H, 1) );
    cudaDeviceSynchronize();

    write_ppm("output.ppm", W, H, reinterpret_cast<const char*>(HostVector<uchar3>(imgData).data()));

    return 0;
}
