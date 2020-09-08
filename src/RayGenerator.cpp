#include <optix_helpers/RayGenerator.h>

namespace optix_helpers {

RayGeneratorObj::RayGeneratorObj(const Buffer& renderBuffer,
                                 const Program& raygenProgram) :
    renderBuffer_(renderBuffer),
    raygenProgram_(raygenProgram)
{
    raygenProgram_->set_buffer(renderBuffer);
}

void RayGeneratorObj::set_pose(const Pose& pose)
{}
 
Buffer RayGeneratorObj::render_buffer() const
{
    return renderBuffer_;
}

Program RayGeneratorObj::raygen_program() const
{
    return raygenProgram_;
}

RayGenerator::RayGenerator() :
    Handle<RayGeneratorObj>()
{}

RayGenerator::RayGenerator(const Buffer& renderBuffer,
                           const Program& raygenProgram) :
    Handle<RayGeneratorObj>(new RayGeneratorObj(renderBuffer, raygenProgram))
{}

};
