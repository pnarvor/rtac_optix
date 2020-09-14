#include <iostream>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac::files;

#include <rtac_base/types/Mesh.h>
#include <rtac_base/types/Pose.h>
using Mesh = rtac::types::Mesh<float,uint32_t,3>;
using Pose = rtac::types::Pose<float>;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/PinHoleView.h>
using namespace optix_helpers;

#include <optix_helpers/samples/raytypes.h>
#include <optix_helpers/samples/materials.h>
#include <optix_helpers/samples/geometries.h>
#include <optix_helpers/samples/models.h>
#include <optix_helpers/samples/items.h>
#include <optix_helpers/samples/utils.h>
using namespace optix_helpers::samples;

#include "cusamples.h"

const std::string raygenSource = R"(

#include <optix.h>

using namespace optix;

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);
rtTextureSampler<uchar4, 2, cudaReadModeNormalizedFloat> checkerboard_texture;

rtBuffer<float3, 2> renderBuffer;

RT_PROGRAM void texture_test()
{
    size_t2 s = renderBuffer.size();
    float2 uv = make_float2(((float)launchIndex.x) / s.x,
                            ((float)launchIndex.y) / s.y);
    float4 c = tex2D(checkerboard_texture, 2.0f*uv.x, 2.0f*uv.y);
    renderBuffer[launchIndex] = make_float3(c.x,c.y,c.z);
}

)";

int main()
{
    int W = 512, H = 512;

    Context context;
    
    TextureSampler checkerboard = textures::checkerboard(context, "checkerboard_texture",
                                                         {0,255,0}, {0,50,255},
                                                         4, 4);

    Buffer renderBuffer = context->create_buffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, "renderBuffer");
    (*renderBuffer)->setSize(W,H);

    Program raygenProgram = context->create_program(Source(raygenSource, "texture_test"));
    raygenProgram->set_object(renderBuffer);
    raygenProgram->set_object(checkerboard);
    
    (*context)->setRayGenerationProgram(0, *raygenProgram);
    //(*context)->setMissProgram(0, *raytypes::RGB::black_miss_program(context));

    (*context)->launch(0,W,H);
    
    utils::display(renderBuffer);

    return 0;
}

