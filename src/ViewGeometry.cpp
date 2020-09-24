#include <optix_helpers/ViewGeometry.h>

#include <rtac_base/algorithm.h>

namespace optix_helpers {

ViewGeometryObj::ViewGeometryObj(const Context& context,
                                 const Buffer& renderBuffer,
                                 const RayType& rayType,
                                 const Source& raygenSource,
                                 const Sources& additionalHeaders) :
    renderBuffer_(renderBuffer),
    raygenProgram_(context->create_program(raygenSource,
        Sources({rayType->definition()}) + additionalHeaders))
{
    raygenProgram_->set_object(renderBuffer);
}

void ViewGeometryObj::update_geometry()
{
}

void ViewGeometryObj::set_pose(const Pose& pose)
{
    pose_ = pose;
}
 
void ViewGeometryObj::set_size(size_t width, size_t height)
{
    renderBuffer_->set_size(width, height);
    this->update_geometry();
}

void ViewGeometryObj::set_range(float zNear, float zFar)
{
    //range modifier
}

void ViewGeometryObj::look_at(const Vector3& target)
{
    this->look_at(target, pose_.translation(), {0.0,0.0,1.0});
}

void ViewGeometryObj::look_at(const Vector3& target,
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

    this->set_pose(Pose::from_rotation_matrix(r, position));
}
 
Buffer ViewGeometryObj::render_buffer() const
{
    return renderBuffer_;
}

Program ViewGeometryObj::raygen_program() const
{
    return raygenProgram_;
}

ViewGeometryObj::Shape ViewGeometryObj::render_shape() const
{
    return renderBuffer_->shape();
}

ViewGeometryObj::Pose ViewGeometryObj::pose() const
{
    return pose_;
}

void ViewGeometryObj::write_data(uint8_t* dest) const
{
    size_t W,H;
    (*renderBuffer_)->getSize(W,H);
    std::memcpy(dest, static_cast<uint8_t*>((*renderBuffer_)->map()),
                W*H*(*renderBuffer_)->getElementSize());
    (*renderBuffer_)->unmap();
}

//ViewGeometry::ViewGeometry() :
//    Handle<ViewGeometryObj>()
//{}
//
//ViewGeometry::ViewGeometry(const Buffer& renderBuffer,
//                           const Program& raygenProgram,
//                           const Pose& pose) :
//    Handle<ViewGeometryObj>(new ViewGeometryObj(renderBuffer, raygenProgram, pose))
//{}
//
//ViewGeometry::ViewGeometry(const std::shared_ptr<ViewGeometryObj>& p) :
//    Handle<ViewGeometryObj>(p)
//{}

};
