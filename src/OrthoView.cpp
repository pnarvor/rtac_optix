#include <optix_helpers/OrthoView.h>

#include <optix_helpers/utils.h>

namespace optix_helpers {

const Source OrthoViewObj::rayGeometryDefinition = Source(R"(
#include <optix.h>
#include <optix_math.h>

rtDeclareVariable(float3, orthoTopLeft,,);
rtDeclareVariable(float3, orthoStepX,,);
rtDeclareVariable(float3, orthoStepY,,);
rtDeclareVariable(float3, orthoDirection,,);
rtDeclareVariable(float2, range,,);

__device__
optix::Ray ortho_ray(const uint2& launchIndex)
{
    return optix::Ray(orthoTopLeft + launchIndex.x*orthoStepX + launchIndex.y*orthoStepY,
                      orthoDirection,
                      range.x, range.y);
}

)", "view/ortho.h");

using namespace rtac::types::indexing;


OrthoViewObj::OrthoViewObj(const Buffer& renderBuffer,
             const Program& raygenProgram,
             const Bounds& bounds,
             const Pose& pose) :
    ViewGeometryObj(renderBuffer, raygenProgram, pose),
    bounds_(bounds)
{
    this->set_range(0.0, RT_DEFAULT_MAX);
}

void OrthoViewObj::update_device_geometry()
{
    size_t w, h;
    (*renderBuffer_)->getSize(w, h);
    float xres = (bounds_.right - bounds_.left)   / w;
    float yres = (bounds_.top   - bounds_.bottom) / h;
    
    Matrix3 R = pose_.rotation_matrix();
    Vector3 imgx = R(all,0);
    Vector3 imgy = R(all,2);

    //using namespace std;
    //cout << yres << endl;
    //cout << imgy.transpose() << endl;

    
    // We take center of pixels for ray origin.
    Vector3 topLeft = pose_.translation()
                    + (bounds_.left + 0.5*xres)*imgx
                    + (bounds_.top  - 0.5*yres)*imgy;
    (*raygenProgram_)["orthoTopLeft"]->setFloat(make_float3(topLeft));
    (*raygenProgram_)["orthoStepX"]->setFloat(make_float3(xres*imgx));
    (*raygenProgram_)["orthoStepY"]->setFloat(make_float3(-yres*imgy));
    (*raygenProgram_)["orthoDirection"]->setFloat(make_float3(-R(all,1)));
}

void OrthoViewObj::set_pose(const Pose& pose)
{
    this->ViewGeometryObj::set_pose(pose);
    this->update_device_geometry();
}

void OrthoViewObj::set_size(size_t width, size_t height)
{
    this->ViewGeometryObj::set_size(width, height);
    this->update_device_geometry();
}

void OrthoViewObj::set_range(float zNear, float zFar)
{
    (*raygenProgram_)["range"]->setFloat(optix::make_float2(zNear, zFar));
}

void OrthoViewObj::set_bounds(const Bounds& bounds)
{
    bounds_ = bounds;
    this->update_device_geometry();
}

const Source& OrthoView::rayGeometryDefinition = OrthoViewObj::rayGeometryDefinition;

OrthoView::OrthoView() :
    Handle<OrthoViewObj>()
{}

OrthoView::OrthoView(const Buffer& renderBuffer,
                     const Program& raygenProgram,
                     const Bounds& bounds,
                     const Pose& pose) :
    Handle<OrthoViewObj>(new OrthoViewObj(renderBuffer, raygenProgram, bounds, pose))
{}

}; //namespace optix_helpers
