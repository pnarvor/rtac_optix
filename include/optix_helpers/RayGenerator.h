#ifndef _DEF_OPTIX_HELPERS_RAY_GENERATOR_H_
#define _DEF_OPTIX_HELPERS_RAY_GENERATOR_H_

#include <iostream>
#include <cstring>

#include <optixu/optixpp.h>

#include <rtac_base/types/Pose.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Buffer.h>
#include <optix_helpers/Program.h>

namespace optix_helpers {

class RayGeneratorObj
{
    public:

    using Pose    = rtac::types::Pose<float>;
    using Vector3 = rtac::types::Vector3<float>;
    using Matrix3 = rtac::types::Matrix3<float>;
    using Shape   = BufferObj::Shape;

    protected:
    
    Buffer  renderBuffer_;  // buffer where the image will be renderered.
    Program raygenProgram_; // compiled ray generation program.
    Pose    pose_;
    
    virtual void update_geometry();
    
    public:

    RayGeneratorObj(const Context& context,
                    const Buffer& renderBuffer,
                    const RayType& rayType,
                    const Source& raygenSource,
                    const Sources& additionalHeaders);
    
    // virtual member function to be reimplemented by subclassing.
    virtual void set_pose(const Pose& pose);
    virtual void set_size(size_t width, size_t height);
    virtual void set_range(float zNear, float zFar);

    void look_at(const Vector3& target);
    void look_at(const Vector3& target,
                 const Vector3& position,
                 const Vector3& up = {0.0,0.0,1.0});

    Buffer  render_buffer()  const;
    Program raygen_program() const;
    Shape   render_shape() const;
    Pose    pose()           const;
    void    write_data(uint8_t* dest) const;
};
using RayGenerator = Handle<RayGeneratorObj>;


}; //namespace optix helpers


#endif //_DEF_OPTIX_HELPERS_RAY_GENERATOR_H_
