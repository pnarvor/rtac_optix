#include <iostream>
using namespace std;

#include <optix_helpers/samples/scenes.h>
#include <optix_helpers/samples/utils.h>
using namespace optix_helpers::samples;

int main()
{
    int W = 1920, H = 1080;
    scenes::Scene0 scene(W,H);
    scene.launch();
    utils::display(scene.view()->render_buffer());

    return 0;
}


