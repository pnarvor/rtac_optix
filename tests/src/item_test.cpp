#include <iostream>
using namespace std;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/Material.h>
#include <optix_helpers/Model.h>
#include <optix_helpers/RayGenerator.h>
#include <optix_helpers/SceneItem.h>
using namespace optix_helpers;

#include <rtac_tools/files.h>
using rtac::files::write_pgm;

#include "cusamples.h"

int main()
{
    //int W = 32, H = 16;
    //int W = 64, H = 32;
    int W = 512, H = 512;
    Context context;

    RayType rayType0 = context->create_raytype(Source(cusample::coloredRay, "colored_ray.h"));
    //cout << rayType0 << endl;

    auto renderer = context->create_raygenerator(W,H,1);
    renderer->set_raygen_program(context->create_raygen_program(
        "render_buffer", Source(cusample::rayGenOrtho, "ortho_z"), {rayType0->definition()}));
    renderer->set_miss_program(context->create_program(
        Source(cusample::coloredMiss, "black_miss"), {rayType0->definition()}));

    context->context()->setEntryPointCount(1);
    context->context()->setRayGenerationProgram(0, renderer->raygen_program());
    context->context()->setMissProgram(0, renderer->miss_program());
    optix::Group topObject = context->context()->createGroup();
    topObject->setAcceleration(context->context()->createAcceleration("Trbvh"));
    renderer->raygen_program()["topObject"]->set(topObject);

    Material white(context->create_material());
    white->add_closest_hit_program(rayType0,
        context->create_program(Source(cusample::whiteMaterial, "closest_hit_white"),
                                {rayType0->definition()}));
    //cout << white->get_closest_hit_program(rayType0) << endl;

    Model sphere0 = context->create_model();
    sphere0->set_geometry(context->create_geometry(
        context->create_program(Source(cusample::sphere, "intersection")),
        context->create_program(Source(cusample::sphere, "bounds"))));
    sphere0->add_material(white);
    
    SceneItem item0 = context->create_scene_item(sphere0);
    item0->set_acceleration(context->context()->createAcceleration("Trbvh"));
    float pose[16] = {1.0,0.0,0.0,0.0,
                      0.0,1.0,0.0,0.0,
                      0.0,0.0,1.0,0.0,
                      0.0,0.0,0.0,1.0};
    item0->set_pose(pose);

    topObject->addChild(item0->node());

    context->context()->launch(0,W,H);

    float* data = (float*)renderer->render_buffer()->map();
    std::vector<uint8_t> imgData(renderer->shape().size());
    for(int i = 0; i < imgData.size(); i++) {
        imgData[i] = 255*data[i];
    }
    //std::ostringstream oss;
    //for(int h = 0; h < H; h++) {
    //    for(int w = 0; w < W; w++) {
    //        //cout << (int)(255.0*data[h*W + w]) << " ";
    //        oss << (int)data[h*W + w];
    //    }
    //    oss << "\n";
    //}
    //cout << oss.str() << endl;
    renderer->render_buffer()->unmap();
    write_pgm("out.pgm", renderer->shape().width, renderer->shape().height, (const char*)imgData.data());


    return 0;
}

