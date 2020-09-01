#include <optix_helpers/RayGenerator.h>

namespace optix_helpers {

RayGeneratorObj::Shape::Shape(size_t w, size_t h, size_t d) :
    width(w), height(h), depth(d)
{}

RayGeneratorObj::Shape::Shape(const std::initializer_list<size_t>& dims) :
    Shape()
{
    auto d = dims.begin();
    if(d == dims.end()) 
        return;
    width = *d;
    d++;
    if(d == dims.end()) 
        return;
    height = *d;
    d++;
    if(d == dims.end()) 
        return;
    depth = *d;
}

size_t RayGeneratorObj::Shape::size() const
{
    return width*height*depth;
}

RayGeneratorObj::RayGeneratorObj(size_t width, size_t height, size_t depth,
                                 const optix::Buffer& buffer,
                                 const RayGenerationProgram& raygen,
                                 const Program& miss) :
    shape_(width, height, depth),
    renderBuffer_(buffer),
    raygenProgram_(raygen),
    missProgram_(miss)
{
    this->update_buffer_size();
    this->update_buffer_variable();
} 
 
void RayGeneratorObj::update_buffer_size()
{
    if(!renderBuffer_) 
        return;
    switch(shape_.depth) {
        case 1:
            renderBuffer_->setFormat(RT_FORMAT_FLOAT);
            break;
        case 2:
            renderBuffer_->setFormat(RT_FORMAT_FLOAT2);
            break;
        case 3:
            renderBuffer_->setFormat(RT_FORMAT_FLOAT3);
            break;
        case 4:
            renderBuffer_->setFormat(RT_FORMAT_FLOAT4);
            break;
        default:
            throw std::runtime_error("Invalid buffer depth (should be [1-4])");
    }
    renderBuffer_->setSize(shape_.width, shape_.height);
}

void RayGeneratorObj::update_buffer_variable()
{
    if(raygenProgram_)
        raygenProgram_->set_render_buffer(renderBuffer_);
}

void RayGeneratorObj::set_raygen_program(const RayGenerationProgram& program)
{
    raygenProgram_ = program;
    this->update_buffer_variable();
}

void RayGeneratorObj::set_miss_program(const Program& program)
{
    missProgram_ = program;
}

optix::Buffer RayGeneratorObj::render_buffer() const
{
    return renderBuffer_;
}

RayGenerationProgram RayGeneratorObj::raygen_program() const
{
    return raygenProgram_;
}

Program RayGeneratorObj::miss_program() const
{
    return missProgram_;
}

RayGenerator::RayGenerator() :
    Handle<RayGeneratorObj>()
{}

RayGenerator::RayGenerator(size_t width, size_t height, size_t depth,
                           const optix::Buffer& buffer,
                           const RayGenerationProgram& raygen,
                           const Program& miss) :
    Handle<RayGeneratorObj>(new RayGeneratorObj(width, height, depth, buffer, raygen, miss))
{}

};
