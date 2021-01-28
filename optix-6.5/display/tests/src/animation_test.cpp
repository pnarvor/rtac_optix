#include <iostream>
#include <vector>
#include <chrono>
using namespace std;

#include <rtac_base/time.h>
using FrameCounter = rtac::time::FrameCounter;

#include <rtac_base/types/Pose.h>
using Pose = rtac::types::Pose<float>;
using Quaternion = rtac::types::Quaternion<float>;

#include <optix_helpers/Buffer.h>
#include <optix_helpers/samples/scenes.h>
#include <optix_helpers/display/GLBuffer.h>
using namespace optix_helpers;
using namespace optix_helpers::samples;
using namespace optix_helpers::display;

#include <rtac_display/Display.h>
#include <rtac_display/GLVector.h>
#include <rtac_display/renderers/ImageRenderer.h>
using namespace rtac::display;

int main()
{
    unsigned int W = 1920, H = 1080;

    Display display;
    
    scenes::Scene0<Buffer> scene(W,H);

    scene.view()->look_at({0.0,0.0,0.0}, { 2.0, 5.0, 4.0});
    float dangle = 0.001;
    Pose R({0.0,0.0,0.0}, Quaternion({cos(dangle/2), 0.0, 0.0, sin(dangle/2)}));
    
    auto imageRenderer = ImageRenderer::New();
    display.add_renderer(imageRenderer);
    
    auto vd = scene.render_buffer()->to_device_vector<float>();
    
    FrameCounter counter;
    while(!display.should_close()) {
        scene.view()->set_pose(R * scene.view()->pose());
        
        scene.launch();

        scene.render_buffer()->to_device_vector<float>(vd);
        GLVector<float> glv(vd);
        //imageRenderer->set_image(scene.render_buffer());
        imageRenderer->set_image({W,H}, glv.gl_id());
        display.draw();
        
        cout << counter;
    }
    cout << endl;

    return 0;
}


