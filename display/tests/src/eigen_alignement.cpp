#include <iostream>
using namespace std;

#include <optix_helpers/display/Display.h>
#include <optix_helpers/display/Renderer.h>
//#include <optix_helpers/display/View.h>
using namespace optix_helpers::display;

int main()
{
    cout << "Creating window" << endl << flush;
    Display window;
    cout << "Creating renderer" << endl << flush;
    auto renderer = Renderer::New();
    return 0;
}
