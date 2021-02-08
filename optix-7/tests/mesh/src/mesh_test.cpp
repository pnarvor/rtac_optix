#include <iostream>
using namespace std;

#include <cuda_runtime.h>
#include <optix.h>

#include <rtac_base/cuda/DeviceMesh.h>
using namespace rtac::cuda;

#include <rtac_optix/utils.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/Pipeline.h>
#include <rtac_optix/AccelerationStruct.h>
using namespace rtac::optix;

#include <mesh_test/ptx_files.h>

#include "mesh_test.h"

using RaygenRecord = SbtRecord<RaygenData>;
using MissRecord   = SbtRecord<MissData>;

int main()
{
    unsigned int W = 16, H = 9;
    auto mesh = DeviceMesh<>::cube();
    auto ptxFiles = mesh_test::get_ptx_files();

    cudaFree(0); // no-op to initialize cuda
    OPTIX_CHECK(optixInit());

    Context context;
    Pipeline pipeline(context);
    pipeline.add_module("src/mesh_test.cu", ptxFiles["src/mesh_test.cu"]);
    pipeline.add_raygen_program("__raygen__mesh_test", "src/mesh_test.cu");
    pipeline.add_miss_program("__miss__mesh_test", "src/mesh_test.cu");
    pipeline.link();

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

    AccelerationStruct handle(context);
    handle.build(buildInput, buildOptions);
    
    return 0;
}
