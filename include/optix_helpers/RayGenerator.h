#ifndef _DEF_OPTIX_HELPERS_RAY_GENERATOR_H_
#define _DEF_OPTIX_HELPERS_RAY_GENERATOR_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <rtac_base/types/Pose.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Buffer.h>
#include <optix_helpers/Program.h>

namespace optix_helpers {

class RayGeneratorObj
{
    public:

    using Pose = rtac::types::Pose<float>;

    protected:
    
    Buffer  renderBuffer_;  // buffer where the image will be renderered.
    Program raygenProgram_; // compiled ray generation program.

    public:

    RayGeneratorObj(const Buffer& renderBuffer,
                    const Program& raygenProgram);

    virtual void set_pose(const Pose& pose);

    Buffer  render_buffer() const;
    Program raygen_program() const;
};

class RayGenerator : public Handle<RayGeneratorObj>
{
    public:

    RayGenerator();
    RayGenerator(const Buffer& renderBuffer,
                 const Program& raygenProgram);
};


}; //namespace optix helpers


#endif //_DEF_OPTIX_HELPERS_RAY_GENERATOR_H_
