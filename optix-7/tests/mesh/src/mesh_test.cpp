#include <iostream>
using namespace std;

#include <cuda_runtime.h>
#include <optix.h>

#include <rtac_base/files.h>
using namespace rtac::files;

#include <rtac_base/cuda/utils.h>
#include <rtac_base/cuda/DeviceMesh.h>
using namespace rtac::cuda;

#include <rtac_optix/utils.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/Pipeline.h>
#include <rtac_optix/AccelerationStruct.h>
using namespace rtac::optix;

#include <mesh_test/ptx_files.h>

#include "mesh_test.h"

using RaygenRecord     = SbtRecord<RaygenData>;
using MissRecord       = SbtRecord<MissData>;
using ClosestHitRecord = SbtRecord<ClosestHitData>;

int main()
{
    unsigned int W = 800, H = 600;
    auto mesh = DeviceMesh<>::cube();
    auto ptxFiles = mesh_test::get_ptx_files();

    cudaFree(0); // no-op to initialize cuda
    OPTIX_CHECK(optixInit());

    auto context = Context::Create();
    auto pipeline = Pipeline::Create(context);
    pipeline->add_module("src/mesh_test.cu", ptxFiles["src/mesh_test.cu"]);
    auto raygenProgram = pipeline->add_raygen_program("__raygen__mesh_test", "src/mesh_test.cu");
    auto missProgram   = pipeline->add_miss_program("__miss__mesh_test", "src/mesh_test.cu");

     // Creating closest hit program
     auto closestHitDesc = zero<OptixProgramGroupDesc>();
     closestHitDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
     closestHitDesc.hitgroup.moduleCH = pipeline->module("src/mesh_test.cu");
     closestHitDesc.hitgroup.entryFunctionNameCH = "__closesthit__mesh_test";
     auto closestHitProgram = pipeline->add_program_group(closestHitDesc);
    
    // linking pipeline
    //pipeline->link();

    // Building mesh acceleration structure
    auto buildOptions = zero<OptixAccelBuildOptions>();
    buildOptions.buildFlags = OPTIX_BUILD_FLAG_NONE;
    buildOptions.operation  = OPTIX_BUILD_OPERATION_BUILD;

    auto buildInput = zero<OptixBuildInput>();
    CUdeviceptr vertexData = reinterpret_cast<CUdeviceptr>(mesh.points().data());
    CUdeviceptr faceData   = reinterpret_cast<CUdeviceptr>(mesh.faces().data());
    const uint32_t inputFlags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    buildInput.triangleArray.numVertices         = mesh.num_points();
    buildInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(DeviceMesh<>::Point);
    buildInput.triangleArray.vertexBuffers       = &vertexData;
    buildInput.triangleArray.numIndexTriplets    = mesh.num_faces();
    buildInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    buildInput.triangleArray.indexStrideInBytes  = sizeof(DeviceMesh<>::Face);
    buildInput.triangleArray.indexBuffer         = faceData;
    buildInput.triangleArray.flags               = inputFlags;
    buildInput.triangleArray.numSbtRecords       = 1;

    auto handle = AccelerationStruct::Create(context);
    handle->build(buildInput, buildOptions);
    
    // Building shader binding table
    auto sbt = zero<OptixShaderBindingTable>();
    RaygenRecord raygenRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(*raygenProgram, &raygenRecord));
    sbt.raygenRecord   = reinterpret_cast<CUdeviceptr>(memcpy::host_to_device(raygenRecord));

    MissRecord missRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(*missProgram, &missRecord));
    sbt.missRecordBase = reinterpret_cast<CUdeviceptr>(memcpy::host_to_device(missRecord));
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    sbt.missRecordCount         = 1;

    ClosestHitRecord chRecord;
    OPTIX_CHECK(optixSbtRecordPackHeader(*closestHitProgram, &chRecord));
    sbt.hitgroupRecordBase          = reinterpret_cast<CUdeviceptr>(memcpy::host_to_device(chRecord));
    sbt.hitgroupRecordStrideInBytes = sizeof(chRecord);
    sbt.hitgroupRecordCount         = 1;

    
    // output buffer
    DeviceVector<uchar3> imgData(W*H);

    auto params = zero<Params>();
    params.width     = W;
    params.height    = H;
    params.imageData = imgData.data();
    params.cam       = samples::PinholeCamera::New({0.0f,0.0f,0.0f}, {5.0f,4.0f,3.0f});
    params.topObject = *handle;
    
    OPTIX_CHECK(optixLaunch(*pipeline, 0,
                            reinterpret_cast<CUdeviceptr>(memcpy::host_to_device(params)),
                            sizeof(params), &sbt, W,H,1));
    cudaDeviceSynchronize();

    HostVector<uchar3> output(imgData);

    write_ppm("output.ppm", W, H, reinterpret_cast<const char*>(output.data()));

    return 0;
}


