#ifndef _DEF_OPTIX_HELPERS_RAY_GENERATOR_H_
#define _DEF_OPTIX_HELPERS_RAY_GENERATOR_H_

#include <iostream>
#include <cstring>

#include <optixu/optixpp.h>

#include <rtac_base/types/Pose.h>
#include <rtac_base/geometry.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Buffer.h>
#include <optix_helpers/Program.h>

namespace optix_helpers {

class RayGenerator
{
    public:

    using Ptr      = Handle<RayGenerator>;
    using ConstPtr = Handle<const RayGenerator>;

    using Pose    = rtac::types::Pose<float>;
    using Vector3 = rtac::types::Vector3<float>;
    using Matrix3 = rtac::types::Matrix3<float>;
    using Shape   = Buffer::Shape;

    protected:
    
    Buffer::Ptr  renderBuffer_;  // buffer where the image will be renderered.
    Program::Ptr raygenProgram_; // compiled ray generation program.
    Pose         pose_;
    
    virtual void update_geometry();
    
    public:

    static Ptr New(const Context::ConstPtr& context,
                   const Buffer::Ptr& renderBuffer,
                   const RayType& rayType,
                   const Source::Ptr& raygenSource,
                   const Sources& additionalHeaders);
    
    RayGenerator(const Context::ConstPtr& context,
                 const Buffer::Ptr& renderBuffer,
                 const RayType& rayType,
                 const Source::Ptr& raygenSource,
                 const Sources& additionalHeaders);
    
    // virtual member function to be reimplemented by subclassing.
    virtual void set_pose(const Pose& pose);
    virtual void set_size(size_t width, size_t height);
    virtual void set_range(float zNear, float zFar);

    void look_at(const Vector3& target);
    void look_at(const Vector3& target,
                 const Vector3& position,
                 const Vector3& up = {0.0,0.0,1.0});

    Buffer::Ptr       render_buffer()  const;
    Program::ConstPtr raygen_program() const;
    Shape             render_shape() const;
    Pose              pose()           const;
    void              write_data(uint8_t* dest) const;
};

}; //namespace optix helpers


#endif //_DEF_OPTIX_HELPERS_RAY_GENERATOR_H_
