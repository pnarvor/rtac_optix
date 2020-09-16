#include <iostream>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac::files;

#include <rtac_base/types/Mesh.h>
#include <rtac_base/types/Pose.h>
using Mesh = rtac::types::Mesh<float,uint32_t,3>;
using Pose = rtac::types::Pose<float>;
using Quaternion = rtac::types::Quaternion<float>;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/PinHoleView.h>
using namespace optix_helpers;

#include <optix_helpers/samples/raytypes.h>
#include <optix_helpers/samples/materials.h>
#include <optix_helpers/samples/geometries.h>
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

    Ray ray = pinhole_ray(launchIndex, 0);

    rtTrace(topObject, ray, payload);
    //renderBuffer[launchIndex] = payload.color.x;
    renderBuffer[launchIndex] = payload.color;
}

)";

int main()
{
    //int W = 16, H = 16;
    //int W = 512, H = 512;
    //int W = 800, H = 600;
    int W = 1920, H = 1080;

    Context context;
    (*context)->setMaxTraceDepth(10);
    (*context)->setMaxCallableProgramDepth(10);
    cout << "Default stack size : " << (*context)->getStackSize() << endl;
    (*context)->setStackSize(8096);
    cout << "Stack size : " << (*context)->getStackSize() << endl;

    raytypes::RGB rayType0(context);
    //cout << rayType0 << endl;

    Material white    = materials::white(context, rayType0);
    Material mirror   = materials::perfect_mirror(context, rayType0);
    Material glass    = materials::perfect_refraction(context, rayType0, 1.1);
    Material lambert  = materials::lambert(context, rayType0, {10.0,10.0,10.0});
    TexturedMaterial checkerboard = materials::checkerboard(context, rayType0, {255,255,255}, {0,0,0}, 10, 10);

    SceneItem square0 = items::square(context,
        {materials::checkerboard(context, rayType0, {0,255,0}, {0,0,255}, 10, 10)},
        10);

    SceneItem cube0 = items::cube(context,
        {materials::checkerboard(context, rayType0, {255,255,255}, {0,0,0}, 4, 4)});
    //SceneItem cube0 = items::cube(context, {mirror});
    //SceneItem cube0 = items::cube(context, {lambert});
    cube0->set_pose(Pose({2.5,0,1}));

    SceneItem cube1 = items::cube(context,
        {materials::checkerboard(context, rayType0, {255,255,255}, {0,0,0}, 4, 4)});
    //SceneItem cube0 = items::cube(context, {mirror});
    cube1->set_pose(Pose({-2.5,4,2}));

    //SceneItem sphere0 = items::sphere(context, {checkerboard});
    //SceneItem sphere0 = items::sphere(context, {white});
    //SceneItem sphere0 = items::sphere(context, {mirror});
    //SceneItem sphere0 = items::sphere(context, {glass});
    //SceneItem sphere0 = items::sphere(context, {lambert});
    //SceneItem sphere0 = items::cube(context, {mirror});
    //SceneItem sphere0 = items::tube(context, {mirror});
    //SceneItem sphere0 = items::tube(context, {glass});
    //SceneItem sphere0 = items::tube(context, {lambert});
    //SceneItem sphere0 = items::tube(context, {checkerboard});
    //SceneItem sphere0 = items::tube(context, {white});
    //SceneItem sphere0 = items::square(context, {lambert});
    //SceneItem sphere0 = items::square(context, {mirror});
    //SceneItem sphere0 = items::parabola(context, {lambert}, .5, 0.0, 2.0);
    SceneItem sphere0 = items::parabola(context, {mirror}, .5, 0.0, 2.0);
    sphere0->set_pose(Pose({0,0,1}));
    //sphere0->set_pose(Pose({0,0.5,1.5}));
    //sphere0->set_pose(Pose({0,0,0}, Quaternion({0.707,-0.707,0,0})));
    
    Model lense = context->create_model();
    //lense->set_geometry(geometries::parabola(context, 0.1, -0.1, 0.1));
    lense->set_geometry(geometries::parabola(context, 0.1, -0.2, 0.2));
    //lense->add_material(mirror);
    //lense->add_material(glass);
    //auto lenseGlass = materials::perfect_refraction(context, rayType0, 2.4);
    auto lenseGlass = materials::perfect_refraction(context, rayType0, 1.7);
    lense->add_material(lenseGlass);

    SceneItem lense0 = context->create_scene_item(lense);
    lense0->set_pose(Pose({0,0,2}));
    SceneItem lense1 = context->create_scene_item(lense);
    lense1->set_pose(Pose({0,0,2}, Quaternion({0.0,1.0,0.0,0.0})));

    Quaternion q({1.0,1.0,0.0,0.0});
    q.normalize();
    SceneItem mirror0 = items::square(context, {mirror});
    mirror0->set_pose(Pose({0,0,0}, q));
    SceneItem mirror1 = items::square(context, {mirror});
    mirror1->set_pose(Pose({0,0,2}, q));

    optix::Group topObject = (*context)->createGroup();
    topObject->setAcceleration((*context)->createAcceleration("Trbvh"));
    topObject->addChild(square0->node());
    topObject->addChild(cube0->node());
    topObject->addChild(cube1->node());
    //topObject->addChild(sphere0->node());
    //topObject->addChild(mirror0->node());
    //topObject->addChild(mirror1->node());
    topObject->addChild(lense0->node());
    topObject->addChild(lense1->node());

    (*mirror->get_closest_hit_program(rayType0))["topObject"]->set(topObject);
    (*glass->get_closest_hit_program(rayType0))["topObject"]->set(topObject);
    (*lenseGlass->get_closest_hit_program(rayType0))["topObject"]->set(topObject);

    Program raygenProgram = context->create_program(Source(raygenSource,"pinhole_test"),
                                                           {rayType0->definition(), PinHoleView::rayGeometryDefinition});
    
    //PinHoleView pinhole(context->create_buffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, "renderBuffer"),
    PinHoleView pinhole(context->create_buffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT3, "renderBuffer"),
                        raygenProgram);
    pinhole->set_size(W,H);
    pinhole->set_range(1.0e-2f, RT_DEFAULT_MAX);
    pinhole->look_at({0.0,0.0,0.0},{ 2.0, 5.0, 4.0});
    //pinhole->look_at({0.0,0.0,0.0},{ 3.0,-3.0, 4.0});
    //pinhole->look_at({0.0,0.0,0.0},{ -5.0,-2.0, 4.0});
    //pinhole->look_at({0.0,0.0,0.0},{-5.0,-4.0,-3.0});
    //pinhole->look_at({0.0,0.0,0.0},{ 5.0, 0.0, 3.0});
    //pinhole->look_at({0.0,1.0,0.0});
    //pinhole->look_at({0.0,0.0,0.0},{ 2.0, 5.0, -4.0});
    pinhole->look_at({0.0,0.0,0.0},{ -1.0, -1.0, 3.5});

    (*context)->setRayGenerationProgram(0, *raygenProgram);
    (*context)->setMissProgram(0, *raytypes::RGB::rgb_miss_program(context, {0.8,0.8,0.8}));

    (*raygenProgram)["topObject"]->set(topObject);
    (*context)->launch(0,W,H);
    
    //utils::display_ascii(pinhole->render_buffer(), 9);
    utils::display(pinhole->render_buffer());

    return 0;
}

