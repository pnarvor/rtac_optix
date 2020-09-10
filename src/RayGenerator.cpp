#include <optix_helpers/RayGenerator.h>

#include <rtac_base/algorithm.h>

namespace optix_helpers {

RayGeneratorObj::RayGeneratorObj(const Buffer& renderBuffer,
                                 const Program& raygenProgram,
                                 const Pose& pose) :
    renderBuffer_(renderBuffer),
    raygenProgram_(raygenProgram),
    pose_(pose)
{
    raygenProgram_->set_buffer(renderBuffer);
}

void RayGeneratorObj::set_pose(const Pose& pose)
{
    pose_ = pose;
}
 
void RayGeneratorObj::set_size(size_t width, size_t height)
{
    (*renderBuffer_)->setSize(width, height);
}

void RayGeneratorObj::set_range(float zNear, float zFar)
{
    //range modifier
}

void RayGeneratorObj::look_at(const Vector3& target)
{
    this->look_at(target, pose_.translation(), {0.0,0.0,1.0});
}

void RayGeneratorObj::look_at(const Vector3& target,
                              const Vector3& position,
                              const Vector3& up)
{
    using namespace rtac::types::indexing;
    // local y points towards target.
    Vector3 y = target - position;
    if(y.norm() < 1e-6) {
        // Camera too close to target, look towards world y.
        y = Vector3({0.0,1.0,0.0});
    }
    y.normalize();

    Vector3 x = y.cross(up);
    if(x.norm() < 1e-6) {
        // We are looking towards up. Using last image up direction.
        x = y.cross(pose_.rotation_matrix()(all,1));
        if(x.norm() < 1e-6) {
            // No luck... We have to find another non-colinear vector.
            x = rtac::algorithm::find_orthogonal(y);
        }
    }
    x.normalize();
    Vector3 z = x.cross(y);

    rtac::types::Matrix3<float> r;
    r(all,0) = x; r(all,1) = y; r(all,2) = z;

    this->set_pose(Pose(position, r));
}
 
Buffer RayGeneratorObj::render_buffer() const
{
    return renderBuffer_;
}

Program RayGeneratorObj::raygen_program() const
{
    return raygenProgram_;
}

RayGeneratorObj::Pose RayGeneratorObj::pose() const
{
    return pose_;
}

RayGenerator::RayGenerator() :
    Handle<RayGeneratorObj>()
{}

RayGenerator::RayGenerator(const Buffer& renderBuffer,
                           const Program& raygenProgram,
                           const Pose& pose) :
    Handle<RayGeneratorObj>(new RayGeneratorObj(renderBuffer, raygenProgram, pose))
{}

};
