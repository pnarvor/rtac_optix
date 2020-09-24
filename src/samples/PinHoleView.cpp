#include <optix_helpers/samples/PinHoleView.h>

#include <cmath>
#include <algorithm>

#include <optix_helpers/utils.h>

namespace optix_helpers { namespace samples { namespace viewgeometries {

const Source PinHoleViewObj::rayGeometryDefinition = Source(R"(
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

PinHoleViewObj::PinHoleViewObj(const Context& context, 
                               const Buffer& renderBuffer,
                               const RayType& rayType,
                               float fovy,
                               const Source& raygenSource,
                               const Sources& additionalHeaders) :
    ViewGeometryObj(context, renderBuffer, rayType, raygenSource,
                    Sources({rayGeometryDefinition}) + additionalHeaders),
    fovy_(fovy)
{
    this->set_range(1.0e-4f, RT_DEFAULT_MAX);
}

void PinHoleViewObj::update_geometry()
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

void PinHoleViewObj::set_pose(const Pose& pose)
{
    this->ViewGeometryObj::set_pose(pose);
    this->update_geometry();
}

void PinHoleViewObj::set_range(float zNear, float zFar)
{
    (*raygenProgram_)["pinholeRange"]->setFloat(optix::make_float2(zNear, zFar));
}

void PinHoleViewObj::set_fovy(float fovy)
{
    fovy_ = fovy;
    this->update_geometry();
}

const Source& PinHoleView::rayGeometryDefinition = PinHoleViewObj::rayGeometryDefinition;

PinHoleView::PinHoleView() :
    Handle<PinHoleViewObj>()
{}

PinHoleView::PinHoleView(const Context& context, 
                         const Buffer& renderBuffer,
                         const RayType& rayType,
                         float fovy,
                         const Source& raygenSource,
                         const Sources& additionalHeaders) :
    Handle<PinHoleViewObj>(new PinHoleViewObj(
        context, renderBuffer, rayType, fovy, raygenSource, additionalHeaders))
{}

PinHoleView::operator ViewGeometry()
{
    return ViewGeometry(std::dynamic_pointer_cast<ViewGeometryObj>(this->obj_));
}

}; //namespace viewgeometries
}; //namespace samples
}; //namespace optix_helpers
