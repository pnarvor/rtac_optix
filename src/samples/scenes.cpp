#include <optix_helpers/samples/scenes.h>

namespace optix_helpers { namespace samples { namespace scenes {

const Source Scene0::raygenSource = Source(R"(
#include <optix.h>

using namespace optix;

#include <rays/RGB.h>
#include <view/pinhole.h>

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);
rtDeclareVariable(rtObject, topObject,,);

rtBuffer<float3, 2> renderBuffer;

RT_PROGRAM void pinhole_scene0()
{
    raytypes::RGB payload;
    payload.color = make_float3(0.0f,0.0f,0.0f);

    Ray ray = pinhole_ray(launchIndex, 0);

    rtTrace(topObject, ray, payload);
    renderBuffer[launchIndex] = payload.color;
}
)", "pinhole_scene0");

Scene0::Scene0(size_t width, size_t height,
               unsigned int glboId)
{
    using namespace std;
    size_t W = width;
    size_t H = height;

    (*context_)->setMaxTraceDepth(10);
    (*context_)->setMaxCallableProgramDepth(10);
    cout << "Default stack size : " << (*context_)->getStackSize() << endl;
    (*context_)->setStackSize(8096);
    cout << "Stack size : " << (*context_)->getStackSize() << endl;

    raytypes::RGB rayType0(context_);

    Material mirror   = materials::perfect_mirror(context_, rayType0);
    Material glass    = materials::perfect_refraction(context_, rayType0, 1.1);
    TexturedMaterial checkerboard = materials::checkerboard(context_, rayType0, 
                                                            {255,255,255},
                                                            {0,0,0}, 10, 10);

    SceneItem square0 = items::square(context_,
        {materials::checkerboard(context_, rayType0, {0,255,0}, {0,0,255}, 10, 10)},
        10);

    SceneItem cube0 = items::cube(context_,
        {materials::checkerboard(context_, rayType0, {255,255,255}, {0,0,0}, 4, 4)});
    cube0->set_pose(Pose({4,0,1}));

    SceneItem cube1 = items::cube(context_,
        {materials::checkerboard(context_, rayType0, {255,255,255}, {0,0,0}, 4, 4)});
    cube1->set_pose(Pose({-2.5,4,2}));

    SceneItem sphere0 = items::sphere(context_, {mirror}, 2.0);
    sphere0->set_pose(Pose({0,-0.5,1}));

    Model lense = context_->create_model();
    lense->set_geometry(geometries::parabola(context_, 0.1, -0.2, 0.2));
    auto lenseGlass = materials::perfect_refraction(context_, rayType0, 1.7);
    lense->add_material(lenseGlass);
    
    SceneItem lense0 = context_->create_scene_item(lense);
    lense0->set_pose(Pose({3,-2,2},
                          Quaternion({cos(0.5), 0.707*sin(0.5), 0.707*sin(0.5), 0.0})));
    SceneItem lense1 = context_->create_scene_item(lense);
    lense1->set_pose(lense0->pose()*Quaternion({0.0,1.0,0.0,0.0}));

    optix::Group topObject = (*context_)->createGroup();
    topObject->setAcceleration((*context_)->createAcceleration("Trbvh"));
    topObject->addChild(square0->node());
    topObject->addChild(cube0->node());
    topObject->addChild(cube1->node());
    topObject->addChild(sphere0->node());
    topObject->addChild(lense0->node());
    topObject->addChild(lense1->node());

    (*mirror->get_closest_hit_program(rayType0))["topObject"]->set(topObject);
    (*glass->get_closest_hit_program(rayType0))["topObject"]->set(topObject);
    (*lenseGlass->get_closest_hit_program(rayType0))["topObject"]->set(topObject);

    Program raygenProgram = context_->create_program(raygenSource,
                                                     {rayType0->definition(),
                                                      PinHoleView::rayGeometryDefinition});
    
    Buffer renderBuffer;
    if(glboId > 0) {
        renderBuffer = context_->create_gl_buffer(RT_BUFFER_OUTPUT,
                                                  RT_FORMAT_FLOAT3,
                                                  glboId,
                                                  "renderBuffer");
    }
    else {
        renderBuffer = context_->create_buffer(RT_BUFFER_OUTPUT,
                                               RT_FORMAT_FLOAT3,
                                               "renderBuffer");
    }
    view_ = PinHoleView(renderBuffer, raygenProgram);
    view_->set_size(W,H);
    view_->set_range(1.0e-2f, RT_DEFAULT_MAX);
    view_->look_at({0.0,0.0,0.0},{ 2.0, 5.0, 4.0});

    (*context_)->setRayGenerationProgram(0, *raygenProgram);
    (*context_)->setMissProgram(0, *raytypes::RGB::rgb_miss_program(context_, {0.8,0.8,0.8}));

    (*raygenProgram)["topObject"]->set(topObject);
}

RayGenerator Scene0::view()
{
    return view_;
}

void Scene0::launch()
{
    size_t W, H;
    (*view_->render_buffer())->getSize(W,H);
    (*context_)->launch(0,W,H);
}

}; //namespace scenes
}; //namespace samples
}; //namespace optix_helpers


