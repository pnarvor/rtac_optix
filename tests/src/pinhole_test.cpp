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

#include <rays/RGB.h>
#include <view/pinhole.h>

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);
rtDeclareVariable(rtObject, topObject,,);

//rtBuffer<float, 2> renderBuffer;
rtBuffer<float3, 2> renderBuffer;

RT_PROGRAM void pinhole_test()
{
    raytypes::RGB payload;
    payload.color = make_float3(0.0f,0.0f,0.0f);

    Ray ray = pinhole_ray(launchIndex);

    rtTrace(topObject, ray, payload);
    //renderBuffer[launchIndex] = payload.color.x;
    renderBuffer[launchIndex] = payload.color;
}

)";

int main()
{
    //int W = 16, H = 16;
    //int W = 512, H = 512;
    int W = 800, H = 600;

    Context context;
    
    raytypes::RGB rayType0(context);
    cout << rayType0 << endl;

    SceneItem cube0 = items::cube(context, rayType0, 0.5, {0.2,0.0,0.0});
    cube0->set_pose(Pose({0.0,0.0,0.9}));

    SceneItem sphere0 = items::sphere(context, rayType0, 0.5);

    optix::Group topObject = (*context)->createGroup();
    topObject->setAcceleration((*context)->createAcceleration("Trbvh"));
    topObject->addChild(cube0->node());
    topObject->addChild(sphere0->node());

    Program raygenProgram = context->create_program(Source(raygenSource,"pinhole_test"),
                                                           {rayType0->definition(), PinHoleView::rayGeometryDefinition});
    
    //PinHoleView pinhole(context->create_buffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, "renderBuffer"),
    PinHoleView pinhole(context->create_buffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, "renderBuffer"),
                        raygenProgram);
    pinhole->set_size(W,H);
    pinhole->look_at({0.0,0.0,0.0},{5.0,0.0,2.0});
    //pinhole->look_at({0.0,1.0,0.0});

    (*context)->setRayGenerationProgram(0, *raygenProgram);
    (*context)->setMissProgram(0, *raytypes::RGB::black_miss_program(context));

    (*raygenProgram)["topObject"]->set(topObject);
    (*context)->launch(0,W,H);
    
    utils::display_ascii(pinhole->render_buffer(), 9);
    utils::display(pinhole->render_buffer());

    //const float* data = static_cast<const float*>((*pinhole->render_buffer())->map());
    ////for(int h = 0; h < H; h++) {
    ////    for(int w = 0; w < W; w++) {
    ////        int index = 3*(W*h + w);
    ////        //cout << data[index] << " ";
    ////        cout << (int)(100*data[index]) << " ";
    ////        //cout << (int)(data[index]) << " ";
    ////    }
    ////    cout << "\n";
    ////}
    ////cout << endl;
    ////write_pgm("out.pgm", W, H, data);
    ////system("eog out.pgm");
    //write_ppm("out.ppm", W, H, data);
    //system("eog out.ppm");
    //(*pinhole->render_buffer())->unmap();

    return 0;
}

