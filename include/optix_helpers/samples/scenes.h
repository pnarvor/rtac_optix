#ifndef _DEF_OPTIX_HELPERS_SAMPLES_TEST_SCENES_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_TEST_SCENES_H_

#include <iostream>

#include <rtac_base/types/Pose.h>

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/RayGenerator.h>
#include <optix_helpers/PinHoleView.h>

#include <optix_helpers/samples/raytypes.h>
#include <optix_helpers/samples/materials.h>
#include <optix_helpers/samples/geometries.h>
#include <optix_helpers/samples/items.h>
#include <optix_helpers/samples/utils.h>
#include <optix_helpers/samples/raygenerators.h>

namespace optix_helpers { namespace samples { namespace scenes {

template <typename RayGeneratorType>
class SceneBase
{
    protected:

    Context          context_;
    RayGeneratorType raygenerator_;

    public:

    SceneBase() {};
    ViewGeometry view() { return raygenerator_->view(); }

    void launch()
    {
        auto shape = raygenerator_->render_shape();
        (*context_)->launch(0, shape.width, shape.height);
    }
};

template <typename RenderBufferType>
class Scene0 : public SceneBase<raygenerators::RgbCamera<RenderBufferType>>
{
    public:

    using Pose = rtac::types::Pose<float>;
    using Quaternion = rtac::types::Quaternion<float>;

    static const Source raygenSource;

    public:

    Scene0(size_t width, size_t height);

    RenderBufferType render_buffer() { return this->raygenerator_->render_buffer(); }
};


template <typename RenderBufferType>
Scene0<RenderBufferType>::Scene0(size_t width, size_t height)
{
    using namespace std;
    size_t W = width;
    size_t H = height;

    (*this->context_)->setMaxTraceDepth(10);
    (*this->context_)->setMaxCallableProgramDepth(10);
    cout << "Default stack size : " << (*this->context_)->getStackSize() << endl;
    (*this->context_)->setStackSize(8096);
    cout << "Stack size : " << (*this->context_)->getStackSize() << endl;

    raytypes::RGB rayType0(this->context_);

    this->raygenerator_ = raygenerators::RgbCamera<RenderBufferType>(this->context_, rayType0, W, H);
    this->raygenerator_->view_->set_range(1.0e-2f, RT_DEFAULT_MAX);
    this->raygenerator_->view_->look_at({0.0,0.0,0.0},{ 2.0, 5.0, 4.0});
    (*this->context_)->setRayGenerationProgram(0, *this->raygenerator_->raygenProgram_);
    (*this->context_)->setMissProgram(0, *raytypes::RGB::rgb_miss_program(this->context_, {0.8,0.8,0.8}));
    
    Material mirror   = materials::perfect_mirror(this->context_, rayType0);
    Material glass    = materials::perfect_refraction(this->context_, rayType0, 1.1);
    TexturedMaterial checkerboard = materials::checkerboard(this->context_, rayType0, 
                                                            {255,255,255},
                                                            {0,0,0}, 10, 10);

    SceneItem square0 = items::square(this->context_,
        {materials::checkerboard(this->context_, rayType0, {0,255,0}, {0,0,255}, 10, 10)},
        10);

    SceneItem cube0 = items::cube(this->context_,
        {materials::checkerboard(this->context_, rayType0, {255,255,255}, {0,0,0}, 4, 4)});
    cube0->set_pose(Pose({4,0,1}));

    SceneItem cube1 = items::cube(this->context_,
        {materials::checkerboard(this->context_, rayType0, {255,255,255}, {0,0,0}, 4, 4)});
    cube1->set_pose(Pose({-2.5,4,2}));

    SceneItem sphere0 = items::sphere(this->context_, {mirror}, 2.0);
    sphere0->set_pose(Pose({0,-0.5,1}));

    Model lense(this->context_);
    lense->set_geometry(geometries::parabola(this->context_, 0.1, -0.2, 0.2));
    auto lenseGlass = materials::perfect_refraction(this->context_, rayType0, 1.7);
    lense->add_material(lenseGlass);
    
    SceneItem lense0(this->context_, lense);
    lense0->set_pose(Pose({3,-2,2},
                          Quaternion({cos(0.5), 0.707*sin(0.5), 0.707*sin(0.5), 0.0})));
    SceneItem lense1(this->context_, lense);
    lense1->set_pose(lense0->pose()*Quaternion({0.0,1.0,0.0,0.0}));

    optix::Group topObject = (*this->context_)->createGroup();
    topObject->setAcceleration((*this->context_)->createAcceleration("Trbvh"));
    topObject->addChild(square0->node());
    topObject->addChild(cube0->node());
    topObject->addChild(cube1->node());
    topObject->addChild(sphere0->node());
    topObject->addChild(lense0->node());
    topObject->addChild(lense1->node());

    ///// THIS
    //(*mirror->get_closest_hit_program(rayType0))["topObject"]->set(topObject);
    //(*glass->get_closest_hit_program(rayType0))["topObject"]->set(topObject);
    //(*lenseGlass->get_closest_hit_program(rayType0))["topObject"]->set(topObject);
    //(*raygenProgram)["topObject"]->set(topObject);

    // CAN BE REPLACED WITH THIS (thanks to optix variable scope system)
    (*this->context_)["topObject"]->set(topObject);
}

template <typename RenderBufferType>
const Source Scene0<RenderBufferType>::raygenSource = Source(R"(
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

}; //namespace scenes
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_TEST_SCENES_H_
