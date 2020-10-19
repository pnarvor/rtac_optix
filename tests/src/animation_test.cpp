#include <iostream>
#include <vector>
#include <chrono>
using namespace std;

#include <rtac_base/misc.h>
using FrameCounter = rtac::misc::FrameCounter;

#include <rtac_base/types/Pose.h>
using Pose = rtac::types::Pose<float>;
using Quaternion = rtac::types::Quaternion<float>;

#include <optix_helpers/samples/scenes.h>
#include <optix_helpers/display/RenderBufferGL.h>
using namespace optix_helpers::samples;
using namespace optix_helpers::display;

#include <rtac_display/Display.h>
#include <rtac_display/ImageRenderer.h>
using namespace rtac::display;

int main()
{
    int W = 1920, H = 1080;

    Display display;
    
    scenes::Scene0<RenderBufferGL> scene(W,H);

    scene.view()->look_at({0.0,0.0,0.0}, { 2.0, 5.0, 4.0});
    float dangle = 0.001;
    Pose R({0.0,0.0,0.0}, Quaternion({cos(dangle/2), 0.0, 0.0, sin(dangle/2)}));
    
    auto imageRenderer = ImageRenderer::New();
    display.add_renderer(imageRenderer);
    
    FrameCounter counter;
    while(!display.should_close()) {
        scene.view()->set_pose(R * scene.view()->pose());
        
        scene.launch();
        imageRenderer->set_image(scene.render_buffer());
        display.draw();
        
        cout << counter;
    }
    cout << endl;

    return 0;
}


