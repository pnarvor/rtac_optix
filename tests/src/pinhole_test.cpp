#include <iostream>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac::files;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/PinHoleView.h>
using namespace optix_helpers;

#include "cusamples.h"

const std::string raygenSource = R"(

#include <optix.h>

using namespace optix;

#include <colored_ray.h>
#include <view/pinhole.h>

rtBuffer<float, 2> renderBuffer;
rtDeclareVariable(uint2, launchIndex,rtLaunchIndex,);

RT_PROGRAM void pinhole_test()
{
    Ray ray = pinhole_ray(launchIndex);
}

)";

int main()
{
    int W = 24, H = 16;

    Context context;
    (*context)->setEntryPointCount(1);
    
    RayType rayType0 = context->create_raytype(Source(cusample::coloredRay, "colored_ray.h"));
    cout << rayType0 << endl;

    optix::Group topObject = (*context)->createGroup();
    topObject->setAcceleration((*context)->createAcceleration("Trbvh"));

    Material white(context->create_material());
    white->add_closest_hit_program(rayType0,
        context->create_program(Source(cusample::whiteMaterial, "closest_hit_white"),
                                {rayType0->definition()}));

    Program raygenProgram = context->create_program(Source(raygenSource,"pinhole_test"),
                                                           {rayType0->definition(), PinHoleView::rayGeometryDefinition});
    
    PinHoleView pinhole(context->create_buffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, "renderBuffer"),
                      raygenProgram);
    pinhole->set_size(W,H);
    pinhole->look_at({0.0,0.0,0.0},{5.0,4.0,3.0});

    (*context)->setRayGenerationProgram(0, *raygenProgram);
    (*context)->setMissProgram(0, *context->create_program(Source(cusample::coloredMiss, "black_miss"),
                                                           {rayType0->definition()}));

    (*context)->launch(0,W,H);

    const float* data = static_cast<const float*>((*pinhole->render_buffer())->map());
    //for(int h = 0; h < H; h++) {
    //    for(int w = 0; w < W; w++) {
    //        //cout << data[h*W + w] << " ";
    //        cout << (int)(10*data[h*W + w]) << " ";
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

