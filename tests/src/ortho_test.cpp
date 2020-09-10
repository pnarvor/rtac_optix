#include <iostream>
using namespace std;

#include <rtac_base/files.h>
using namespace rtac::files;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
#include <optix_helpers/OrthoView.h>
using namespace optix_helpers;

#include "cusamples.h"

const std::string raygenSource = R"(

#include <optix.h>

using namespace optix;

#include <colored_ray.h>
#include <view/ortho.h>

rtBuffer<float, 2> renderBuffer;
rtDeclareVariable(uint2, launchIndex,rtLaunchIndex,);

RT_PROGRAM void ortho_test()
{
    Ray ray = ortho_ray(launchIndex);

    //renderBuffer[launchIndex] = ray.origin.x;
    //renderBuffer[launchIndex] = ray.origin.y;
    renderBuffer[launchIndex] = ray.origin.z;

    //renderBuffer[launchIndex] = ray.direction.x;
    //renderBuffer[launchIndex] = ray.direction.y;
    //renderBuffer[launchIndex] = ray.direction.z;

    //renderBuffer[launchIndex] = launchIndex.x;
    //renderBuffer[launchIndex] = launchIndex.y;
}

)";

int main()
{
    int W = 24, H = 16;

    Context context;
    (*context)->setEntryPointCount(1);
    
    RayType rayType0 = context->create_raytype(Source(cusample::coloredRay, "colored_ray.h"));
    cout << rayType0 << endl;

    cout << OrthoView::rayGeometryDefinition << endl;
    Program raygenProgram = context->create_program(Source(raygenSource,"ortho_test"),
                                                           {rayType0->definition(), OrthoView::rayGeometryDefinition});
    
    OrthoView ortho(context->create_buffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT, "renderBuffer"),
                    raygenProgram);
    ortho->set_size(W,H);
    ortho->look_at({2,2,1},{0.0,0.0,0.0});
    //ortho->set_bounds({-100,100,-100,100});

    (*context)->setRayGenerationProgram(0, *raygenProgram);

    (*context)->launch(0,W,H);

    const float* data = static_cast<const float*>((*ortho->render_buffer())->map());
    for(int h = 0; h < H; h++) {
        for(int w = 0; w < W; w++) {
            //cout << data[h*W + w] << " ";
            cout << (int)(10*data[h*W + w]) << " ";
            //cout << (int)(data[h*W + w]) << " ";
        }
        cout << "\n";
    }
    cout << endl;
    write_pgm("out.pgm", W, H, data);
    system("eog out.pgm");
    (*ortho->render_buffer())->unmap();

    return 0;
}

