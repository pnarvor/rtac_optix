#include <optix_helpers/samples/maths.h>


namespace optix_helpers { namespace samples { namespace maths {

const Source::ConstPtr maths = Source::New(R"(

#include <optix.h>
#include <optix_math.h>

// a collection of generic mathematical functions such as quadratic equation solving

__device__
float3 reflection(const float3& v, const float3& n)
{
    float3 p = cross(v,n);
    if(dot(p,p) < 1.0e-6f) {
        return -v;
    }
    p = normalize(cross(n,p));
    return -dot(v,n)*n + dot(v,p)*p;
}

__device__
float3 refraction(const float3& i, const float3& n, float refractionIndex)
{
    float3 p = cross(i,n);
    if(dot(p,p) < 1.0e-6f) {
        return i;
    }
    p = normalize(cross(n,p));
    
    if(dot(i,n) <= 0.0f) {
        // entering material
        float sint2 = dot(i,p) / refractionIndex;
        return -sqrt(1.0f - sint2*sint2)*n + sint2*p;
    }
    else {
        // extiing material
        float sint2 = dot(i,p) * refractionIndex;
        return sqrt(1.0f - sint2*sint2)*n + sint2*p;
    }
}

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

__device__
bool sphere_intersection(const optix::Ray& ray, float radius, float& tmin, float& tmax)
{
    return quadratic_solve(1.0f,
                           2.0f*dot(ray.origin, ray.direction),
                           dot(ray.origin, ray.origin) - radius*radius,
                           tmin, tmax);
}

__device__
float3 sphere_normal(const float3& p)
{
    return normalize(p);
}

__device__
bool tube_intersection(const optix::Ray& ray, float radius,
                       float& tmin, float& tmax)
{
    float rayLengthXY = ray.direction.x*ray.direction.x + ray.direction.y*ray.direction.y;
    if(rayLengthXY < 1.0e-6f)
        return false;
    return quadratic_solve(
        rayLengthXY, 
        2.0f*(ray.origin.x*ray.direction.x + ray.origin.y*ray.direction.y),
        ray.origin.x*ray.origin.x + ray.origin.y*ray.origin.y - radius*radius,
        tmin, tmax);
}

__device__
float3 tube_normal(const float3& p)
{
    return normalize(make_float3(p.x, p.y, 0.0f));
}

__device__
bool parabola_intersection(const optix::Ray& ray, float a, float b,
                           float& tmin, float& tmax)
{
    float2 oxy = make_float2(ray.origin.x, ray.origin.y);
    float2 dxy = make_float2(ray.direction.x, ray.direction.y);
    return quadratic_solve(a*dot(dxy,dxy),
                           2.0f*a*dot(oxy,dxy) - ray.direction.z,
                           a*dot(oxy,oxy) + b - ray.origin.z,
                           tmin, tmax);
}

__device__
float3 parabola_normal(const float3 p, float a, float b)
{
    float3 n;
    float pnormXY = sqrt(p.x*p.x + p.y*p.y);
    if(pnormXY < 1.0e-6f) {
        if(a < 0.0f) {
            n = make_float3(0.0f,0.0f,1.0f);
        }
        else {
            n = make_float3(0.0f,0.0f,-1.0f);
        }
    }
    else {
        n.x = p.x / pnormXY;
        n.y = p.y / pnormXY;
        if(a >= 0.0f)
            n.z = -0.5f / (a*pnormXY);
        else
            n.z = 0.5f / (a*pnormXY);
        n = normalize(n);
    }
    return n;
}
)", "optix_helpers/maths.h");

}; //namespace maths
}; //namespace samples
}; //namespace optix_helpers

