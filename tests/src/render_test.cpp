#include <iostream>
using namespace std;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/Material.h>
#include <optix_helpers/Model.h>
#include <optix_helpers/RayGenerator.h>
using namespace optix_helpers;

#include "cusamples.h"

int main()
{
    //int W = 32, H = 16;
    int W = 64, H = 32;
    Context context;

    RayType rayType0 = context->create_raytype(Source(cusample::coloredRay, "colored_ray.h"));
    //cout << rayType0 << endl;

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

    auto renderer = context->create_raygenerator(W,H,1);
    renderer->set_raygen_program(context->create_raygen_program(
        "render_buffer", Source(cusample::rayGenOrtho, "ortho_z"), {rayType0->definition()}));
    renderer->set_miss_program(context->create_program(
        Source(cusample::coloredMiss, "black_miss"), {rayType0->definition()}));

    context->context()->setEntryPointCount(1);
    context->context()->setRayGenerationProgram(0, renderer->raygen_program());
    context->context()->setMissProgram(0, renderer->miss_program());

    optix::GeometryGroup topObject = context->context()->createGeometryGroup();
    topObject->setAcceleration(context->context()->createAcceleration("Trbvh"));
    topObject->addChild(sphere0);
    context->context()["topObject"]->set(topObject);
    //renderer->raygen_program()["topObject"]->set(topObject);

    context->context()->launch(0,W,H);

    float* data = (float*)renderer->render_buffer()->map();
    for(int h = 0; h < H; h++) {
        for(int w = 0; w < W; w++) {
            //cout << (int)(255.0*data[h*W + w]) << " ";
            cout << (int)data[h*W + w];
        }
        cout << "\n";
    }
    renderer->render_buffer()->unmap();

    return 0;
}

