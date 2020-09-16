#include <optix_helpers/samples/maths.h>


namespace optix_helpers { namespace samples { namespace maths {

const Source maths = Source(R"(

#include <optix.h>
#include <optix_math.h>

// a collection of generic mathematical functions such as quadratic equation solving

__device__
bool quadratic_solve(float a, float b, float c, float& res1, float& res2)
{
    float delta = b*b - 4.0f*a*c;
    if(delta < 0.0f) {
        return false;
    }
    delta = sqrt(delta);
    res1 = 0.5f*(-b - delta) / a;
    res2 = 0.5f*(-b + delta) / a;
    return true;
}

)", "optix_helpers/maths.h");

}; //namespace maths
}; //namespace samples
}; //namespace optix_helpers

