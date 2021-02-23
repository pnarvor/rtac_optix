#include <iostream>
using namespace std;

#include <rtac_base/files.h>

#include <rtac_base/cuda/utils.h>
#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>
using namespace rtac::cuda;

#include <rtac_optix/utils.h>
#include <rtac_optix/Pipeline.h>
#include <rtac_optix/MeshAccelStruct.h>
#include <rtac_optix/InstanceAccelStruct.h>
using namespace rtac::optix;

#include <rtac_optix_7_sbt_indexes0/ptx_files.h>

#include "sbt_indexes0.h"

using RaygenRecord = SbtRecord<RaygenData>;
using MissRecord   = SbtRecord<MissData>;
using HitRecord    = SbtRecord<HitData>;

template<typename T>
void print_output(unsigned int W, unsigned int H, const DeviceVector<T>& output);

int main()
{
    cout << "RayType count    : " << RayBuilder::RayTypeCount            << endl;
    cout << "RGBRay index     : " << RayBuilder::index<RGBRay<uchar3>>() << endl;
    cout << "ShadowRay index  : " << RayBuilder::index<ShadowRay>()      << endl;

    optix_init();
    auto context  = Context::Create();
    auto pipeline = Pipeline::Create(context);
    pipeline->add_module("sbt_indexes0",
                         rtac_optix_7_sbt_indexes0::get_ptx_files()["src/sbt_indexes0.cu"]);
    auto raygen = pipeline->add_raygen_program("__raygen__sbt_indexes0", "sbt_indexes0");
    auto miss   = pipeline->add_miss_program("__miss__sbt_indexes0", "sbt_indexes0");
    auto hitGroupDesc = rtac::optix::zero<OptixProgramGroupDesc>();
    hitGroupDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitGroupDesc.hitgroup.moduleCH            = pipeline->module("sbt_indexes0");
    hitGroupDesc.hitgroup.entryFunctionNameCH = "__closesthit__sbt_indexes0";
    auto hitGroup = pipeline->add_program_group(hitGroupDesc);

    auto cubeMesh = MeshAccelStruct::cube_data();
    auto cube0 = MeshAccelStruct::Create(context, cubeMesh);

    // building a per-triangle material (=array of sbt index offsets)
    cube0->set_sbt_flags(std::vector<unsigned int>(3, OPTIX_GEOMETRY_FLAG_NONE));
    DeviceVector<unsigned char> sbtIndexOffsets(
        std::vector<unsigned char>({0,2,0,2,0,2,0,2,0,2,0,2}));
    cube0->build_input().triangleArray.sbtIndexOffsetBuffer = (CUdeviceptr)sbtIndexOffsets.data();
    cube0->build_input().triangleArray.sbtIndexOffsetSizeInBytes = sizeof(unsigned char);

    auto topObject = InstanceAccelStruct::Create(context);
    auto inst0 = topObject->add_instance(*cube0);
    auto inst1 = topObject->add_instance(*cube0);
    //inst1->set_sbt_offset(200);
    inst1->set_transform({1,0,0,-4,
                          0,1,0, 2,
                          0,0,1, 0});

    auto sbt = rtac::optix::zero<OptixShaderBindingTable>();

    RaygenRecord raygenRecord;
    OPTIX_CHECK( optixSbtRecordPackHeader(*raygen, &raygenRecord) );
    sbt.raygenRecord = (CUdeviceptr)memcpy::host_to_device(raygenRecord);

    std::vector<MissRecord> missRecords(255);
    for(int i = 0; i < missRecords.size(); i++) {
        missRecords[i].data.value = i;
        OPTIX_CHECK( optixSbtRecordPackHeader(*miss, &missRecords[i]) );
    }
    DeviceVector<MissRecord> dMissRecords(missRecords);
    sbt.missRecordBase          = (CUdeviceptr)dMissRecords.data();
    sbt.missRecordCount         = dMissRecords.size();
    sbt.missRecordStrideInBytes = sizeof(MissRecord);
    
    //std::vector<HitRecord> hitRecords(255);
    std::vector<HitRecord> hitRecords(127);
    for(int i = 0; i < hitRecords.size(); i++) {
        hitRecords[i].data.value = hitRecords.size() - 1 - i;
        OPTIX_CHECK( optixSbtRecordPackHeader(*hitGroup, &hitRecords[i]) );
    }
    DeviceVector<HitRecord> dHitRecords(hitRecords);
    sbt.hitgroupRecordBase          = (CUdeviceptr)dHitRecords.data();
    sbt.hitgroupRecordCount         = hitRecords.size();
    sbt.hitgroupRecordStrideInBytes = sizeof(HitRecord);

    //unsigned int W = 32, H = 24;
    //DeviceVector<unsigned char> output(W*H);

    unsigned int W = 800, H = 600;
    DeviceVector<uchar3> output(W*H);

    auto params = rtac::optix::zero<Params>();
    params.width  = W;
    params.height = H;
    params.output = output.data();
    params.cam    = samples::PinholeCamera::New(float3({0.0f,0.0f,0.0f}),
                                                float3({5.0f,4.0f,3.0f}));
    //params.topObject = *cube0;
    params.topObject = *topObject;
    
    OPTIX_CHECK( optixLaunch(*pipeline, 0,
                             (CUdeviceptr)memcpy::host_to_device(params), sizeof(Params),
                             &sbt, W, H, 1) );
    cudaDeviceSynchronize();

    //print_output(W,H,output);

    //HostVector<unsigned char> data(output);
    //rtac::files::write_pgm("output.pgm", W, H, (const char*)data.data());
    HostVector<uchar3> data(output);
    rtac::files::write_ppm("output.ppm", W, H, (const char*)data.data());

    return 0;
}

template<typename T>
void print_output(unsigned int W, unsigned int H, const DeviceVector<T>& output)
{
    HostVector<T> data(output);
    for(int h = 0; h < H; h++) {
        for(int w = 0; w < W; w++) {
            cout << " " << (unsigned int)data[W*h + w];
        }
        cout << endl;
    }
}
