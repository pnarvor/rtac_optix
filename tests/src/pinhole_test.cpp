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
using namespace optix_helpers::samples;

#include "cusamples.h"

const std::string raygenSource = R"(

#include <optix.h>

using namespace optix;

#include <rays/RGB.h>
#include <view/pinhole.h>

rtBuffer<float, 2> renderBuffer;
rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);

rtDeclareVariable(rtObject, topObject,,);

RT_PROGRAM void pinhole_test()
{
    raytypes::RGB payload;
    payload.color = make_float3(0.0f,0.0f,0.0f);

    Ray ray = pinhole_ray(launchIndex);

    rtTrace(topObject, ray, payload);
    renderBuffer[launchIndex] = payload.color.x;

    //renderBuffer[launchIndex] = ray.origin.x;
    //renderBuffer[launchIndex] = ray.origin.y;
    //renderBuffer[launchIndex] = ray.origin.z;
    
    //renderBuffer[launchIndex] = ray.direction.x;
    //renderBuffer[launchIndex] = ray.direction.y;
    //renderBuffer[launchIndex] = ray.direction.z;

}

)";

int main()
{
    //int W = 16, H = 16;
    //int W = 512, H = 512;
    int W = 800, H = 600;

    Context context;
    (*context)->setEntryPointCount(1);
    
    raytypes::RGB rayType0(context);
    cout << rayType0 << endl;

    Material white = materials::white(context, rayType0);

    Model cubeModel = models::cube(context, 0.5);
    cubeModel->add_material(white);
    SceneItem cube0 = context->create_scene_item(cubeModel);
    cube0->set_pose(Pose({0.0,0.0,0.9}));

    Model sphereModel = models::sphere(context, 0.5);
    sphereModel->add_material(white);
    SceneItem sphere0 = context->create_scene_item(sphereModel);

    optix::Group topObject = (*context)->createGroup();
    topObject->setAcceleration((*context)->createAcceleration("Trbvh"));
    topObject->addChild(cube0->node());
    topObject->addChild(sphere0->node());

    Program raygenProgram = context->create_program(Source(raygenSource,"pinhole_test"),
                                                           {rayType0->definition(), PinHoleView::rayGeometryDefinition});
    
    PinHoleView pinhole(context->create_buffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, "renderBuffer"),
                      raygenProgram);
    pinhole->set_size(W,H);
    pinhole->look_at({0.0,0.0,0.0},{5.0,4.0,2.0});
    //pinhole->look_at({0.0,1.0,0.0});

    (*context)->setRayGenerationProgram(0, *raygenProgram);
    (*context)->setMissProgram(0, *raytypes::RGB::black_miss_program(context));

    (*raygenProgram)["topObject"]->set(topObject);
    (*context)->launch(0,W,H);

    const float* data = static_cast<const float*>((*pinhole->render_buffer())->map());
    //for(int h = 0; h < H; h++) {
    //    for(int w = 0; w < W; w++) {
    //        //cout << data[h*W + w] << " ";
    //        cout << (int)(100*data[h*W + w]) << " ";
    //        //cout << (int)(data[h*W + w]) << " ";
    //    }
    //    cout << "\n";
    //}
    //cout << endl;
    write_pgm("out.pgm", W, H, data);
    system("eog out.pgm");
    (*pinhole->render_buffer())->unmap();

    return 0;
}

