#include <optix_helpers/samples/PinHole.h>

#include <cmath>
#include <algorithm>

#include <optix_helpers/utils.h>

namespace optix_helpers { namespace samples { namespace raygenerators {

const Source::Ptr PinHole::rayGeometryDefinition = Source::New(R"(
#include <optix.h>
#include <optix_math.h>

rtDeclareVariable(float3, pinholeOrigin,,);
rtDeclareVariable(float3, pinholeTopLeftDir,,);
rtDeclareVariable(float3, pinholeStepX,,);
rtDeclareVariable(float3, pinholeStepY,,);
rtDeclareVariable(float2, pinholeRange,,);

__device__
optix::Ray pinhole_ray(const uint2& launchIndex, unsigned int rayType)
{
    return optix::Ray(pinholeOrigin,
        normalize(pinholeTopLeftDir + launchIndex.x*pinholeStepX + launchIndex.y*pinholeStepY),
        rayType, pinholeRange.x, pinholeRange.y);
}

)", "view/pinhole.h");

using namespace rtac::types::indexing;

PinHole::Ptr PinHole::New(const Context::ConstPtr& context, 
                          const Buffer::Ptr& renderBuffer,
                          const RayType& rayType,
                          const Source::Ptr& raygenSource,
                          const Sources& additionalHeaders)
{
    return Ptr(new PinHole(context, renderBuffer,rayType, raygenSource, additionalHeaders));
}

PinHole::PinHole(const Context::ConstPtr& context, 
                 const Buffer::Ptr& renderBuffer,
                 const RayType& rayType,
                 const Source::Ptr& raygenSource,
                 const Sources& additionalHeaders) :
    RayGenerator(context, renderBuffer, rayType, raygenSource,
                 Sources({rayGeometryDefinition}) + additionalHeaders),
    fovy_(defaultFovy)
{
    this->set_range(1.0e-4f, RT_DEFAULT_MAX);
}

void PinHole::update_geometry()
{
    size_t w, h;
    (*renderBuffer_)->getSize(w, h);
    
    // Assuming unit distance from camera focal point to a virtual image plane.
    // fovy_ is the field of view in the minimal image dimension.
    // resolution is the physical pixel size of the virtual image plane.
    float resolution = 2.0*std::tan(0.5f*M_PI*fovy_/180.0) / std::min(w,h);

    Matrix3 R = pose_.rotation_matrix();
    // We take center of pixels for ray origin. So topLest is shifted by half a pixel.
    Vector3 topLeft = R*Vector3({-0.5f*resolution*(w - 1), 1.0f, 0.5f*resolution*(h - 1)});
    Vector3 stepX   = R*Vector3({resolution, 0.0, 0.0});
    Vector3 stepY   = R*Vector3({0.0, 0.0, -resolution});
    
    (*raygenProgram_)["pinholeOrigin"]->setFloat(make_float3(pose_.translation()));
    (*raygenProgram_)["pinholeTopLeftDir"]->setFloat(make_float3(topLeft));
    (*raygenProgram_)["pinholeStepX"]->setFloat(make_float3(stepX));
    (*raygenProgram_)["pinholeStepY"]->setFloat(make_float3(stepY));
}

void PinHole::set_range(float zNear, float zFar)
{
    (*raygenProgram_)["pinholeRange"]->setFloat(optix::make_float2(zNear, zFar));
}

void PinHole::set_fovy(float fovy)
{
    fovy_ = fovy;
    this->update_geometry();
}

}; //namespace raygenerators
}; //namespace samples
}; //namespace optix_helpers
