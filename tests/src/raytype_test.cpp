#include <iostream>
using namespace std;

#include <optix_helpers/Context.h>
#include <optix_helpers/RayType.h>
using namespace optix_helpers;

#include <optix_helpers/samples/raytypes.h>
using namespace optix_helpers::samples;


int main()
{
    Context context;
    
    cout << "RGB raytype index : " << raytypes::RGB::index << endl;
    raytypes::RGB raytype0(context);
    cout << "RGB raytype index : " << raytypes::RGB::index << endl;
    cout << "RGB raytype instance index : " << raytype0->index() << endl;

    return 0;
}

