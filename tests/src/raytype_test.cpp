#include <iostream>
using namespace std;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
using namespace optix_helpers;

#include <optix_helpers/samples/raytypes.h>
using namespace optix_helpers::samples;


int main()
{
    auto context = Context::New();
    
    cout << "RGB raytype index : " << raytypes::RGB::typeIndex << endl;
    raytypes::RGB raytype0(context);
    cout << "RGB raytype index : " << raytypes::RGB::typeIndex << endl;
    cout << "RGB raytype instance index : " << raytype0.index() << endl;

    return 0;
}

