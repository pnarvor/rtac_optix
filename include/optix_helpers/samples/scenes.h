#ifndef _DEF_OPTIX_HELPERS_SAMPLES_TEST_SCENES_H_
#define _DEF_OPTIX_HELPERS_SAMPLES_TEST_SCENES_H_

#include <iostream>

#include <rtac_base/types/Pose.h>

//#include <optix_helpers/Context.h>
#include <optix_helpers/Scene.h>

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

    Scene            context_;
    RayGeneratorType raygenerator_;

    public:

    SceneBase() :
        context_(Scene::New())
    {}

    RayGeneratorType view()
    { 
        return raygenerator_;
    }

    void add_child(const SceneItem& item)
    {
        context_->add_child(item);
    }

    void launch()
    {
        auto shape = raygenerator_->render_shape();
        (*context_)->launch(0, shape.width, shape.height);
    }

    Scene context()
    {
        return context_;
    }
};

template <typename RenderBufferType>
class SceneRGB0 : public SceneBase<raygenerators::PinHole>
{
    public:

    using Pose = rtac::types::Pose<float>;
    using Quaternion = rtac::types::Quaternion<float>;

    static const Source raygenSource;

    protected:

    raytypes::RGB raytype_;
    RenderBufferType renderBuffer_;

    public:

    SceneRGB0(size_t width, size_t height);

    RenderBufferType render_buffer()
    {
        return renderBuffer_;
    }

    raytypes::RGB raytype()
    {
        return raytype_;
    }
};

template <typename RenderBufferType>
const Source SceneRGB0<RenderBufferType>::raygenSource = Source(R"(
#include <optix.h>

using namespace optix;

#include <rays/RGB.h>
#include <view/pinhole.h>

rtDeclareVariable(uint2, launchIndex, rtLaunchIndex,);
rtDeclareVariable(rtObject, topObject,,);

rtBuffer<float3, 2> renderBuffer;

RT_PROGRAM void rgb_camera()
{
    raytypes::RGB payload;
    payload.color = make_float3(0.0f,0.0f,0.0f);

    Ray ray = pinhole_ray(launchIndex, 0);

    rtTrace(topObject, ray, payload);
    renderBuffer[launchIndex] = payload.color;
}
)", "rgb_camera");

template <typename RenderBufferType>
SceneRGB0<RenderBufferType>::SceneRGB0(size_t width, size_t height) :
    SceneBase(),
    raytype_(this->context_)
{
    renderBuffer_ = RenderBufferType::New(context_, RT_FORMAT_FLOAT3, "renderBuffer");
    this->raygenerator_ = raygenerators::PinHole::New(context_, renderBuffer_, raytype_, raygenSource);
    this->raygenerator_->set_size(width,height);
    this->raygenerator_->set_range(1.0e-2f, RT_DEFAULT_MAX);
    this->raygenerator_->look_at({0.0,0.0,0.0},{ 2.0, 5.0, 4.0});

    (*this->context_)->setRayGenerationProgram(0, *this->raygenerator_->raygen_program());
    (*this->context_)->setMissProgram(0, *raytypes::RGB::rgb_miss_program(this->context_, {0.8,0.8,0.8}));
}

// 
template <typename RenderBufferType>
class Scene0 : public SceneRGB0<RenderBufferType>
{
    public:

    using Pose = rtac::types::Pose<float>;
    using Quaternion = rtac::types::Quaternion<float>;

    static const Source raygenSource;

    public:

    Scene0(size_t width, size_t height);
};

template <typename RenderBufferType>
Scene0<RenderBufferType>::Scene0(size_t width, size_t height) :
    SceneRGB0<RenderBufferType>(width, height)
{
    using namespace std;
    
    Material mirror   = materials::perfect_mirror(this->context_, this->raytype_);
    Material glass    = materials::perfect_refraction(this->context_, this->raytype_, 1.1);
    TexturedMaterial checkerboard = materials::checkerboard(this->context_, this->raytype_, 
                                                            {255,255,255},
                                                            {0,0,0}, 10, 10);

    SceneItem square0 = items::square(this->context_,
        {materials::checkerboard(this->context_, this->raytype_, {0,255,0}, {0,0,255}, 10, 10)},
        10);

    SceneItem cube0 = items::cube(this->context_,
        {materials::checkerboard(this->context_, this->raytype_, {255,255,255}, {0,0,0}, 4, 4)});
    cube0->set_pose(Pose({4,0,1}));

    SceneItem cube1 = items::cube(this->context_,
        {materials::checkerboard(this->context_, this->raytype_, {255,255,255}, {0,0,0}, 4, 4)});
    cube1->set_pose(Pose({-2.5,4,2}));

    SceneItem sphere0 = items::sphere(this->context_, {mirror}, 2.0);
    sphere0->set_pose(Pose({0,-0.5,1}));

    auto lense = Model::New(this->context_);
    lense->set_geometry(geometries::parabola(this->context_, 0.1, -0.2, 0.2));
    auto lenseGlass = materials::perfect_refraction(this->context_, this->raytype_, 1.7);
    lense->add_material(lenseGlass);
    
    SceneItem lense0(this->context_, lense);
    lense0->set_pose(Pose({3,-2,2},
                          Quaternion({cos(0.5), 0.707*sin(0.5), 0.707*sin(0.5), 0.0})));
    SceneItem lense1(this->context_, lense);
    lense1->set_pose(lense0->pose()*Quaternion({0.0,1.0,0.0,0.0}));

    SceneItem mesh0 = items::mesh(this->context_, rtac::types::Mesh<float, uint32_t>::cube());
    mesh0->model()->add_material(mirror);
    mesh0->set_pose(Pose({-2.3, -4.3, 1.0}));

    this->add_child(square0);
    this->add_child(cube0);
    this->add_child(cube1);
    this->add_child(sphere0);
    this->add_child(lense0);
    this->add_child(lense1);
    this->add_child(mesh0);
}

}; //namespace scenes
}; //namespace samples
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_SAMPLES_TEST_SCENES_H_
