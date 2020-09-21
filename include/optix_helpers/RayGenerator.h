#ifndef _DEF_OPTIX_HELPERS_RAY_GENERATOR_H_
#define _DEF_OPTIX_HELPERS_RAY_GENERATOR_H_

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>

namespace optix_helpers {

template <typename ViewGeometryType, RTformat BufferFormat>
class RayGeneratorObj
{
    public:

    struct BufferShape
    {
        size_t width;
        size_t height;
    };
    
    Buffer           renderBuffer_;
    Program          raygenProgram_;
    ViewGeometryType view_;

    public:
    
    RayGeneratorObj(const Context& context, const RayType& rayType,
                    const Source& raygenSource,
                    const std::string& renderBufferName = "renderBuffer");

    void set_size(size_t width, size_t height);

    ViewGeometryType view() const;
    BufferShape render_shape() const;
};

template <typename ViewGeometryType, RTformat BufferFormat>
using RayGenerator = Handle<RayGeneratorObj<ViewGeometryType, BufferFormat>>;

// implementation
template <typename ViewGeometryType, RTformat BufferFormat>
RayGeneratorObj<ViewGeometryType, BufferFormat>::
RayGeneratorObj(const Context& context, const RayType& rayType,
                const Source& raygenSource, const std::string& renderBufferName) :
    renderBuffer_(context->create_buffer(RT_BUFFER_OUTPUT, BufferFormat, renderBufferName)),
    raygenProgram_(context->create_program(raygenSource,
        {rayType->definition(), ViewGeometryType::rayGeometryDefinition})),
    view_(renderBuffer_, raygenProgram_)
{
    raygenProgram_->set_buffer(renderBuffer_);
}

template <typename ViewGeometryType, RTformat BufferFormat>
void RayGeneratorObj<ViewGeometryType, BufferFormat>::
set_size(size_t width, size_t height)
{
    (*renderBuffer_)->setSize(width, height);
    view_->set_size(width, height);
}

template <typename ViewGeometryType, RTformat BufferFormat>
ViewGeometryType RayGeneratorObj<ViewGeometryType, BufferFormat>::view() const
{
    return view_;
}

template <typename ViewGeometryType, RTformat BufferFormat>
typename RayGeneratorObj<ViewGeometryType, BufferFormat>::BufferShape RayGeneratorObj<ViewGeometryType, BufferFormat>::render_shape() const
{
    BufferShape res;
    (*renderBuffer_)->getSize(res.width, res.height);
    return res;
}


}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_RAY_GENERATOR_H_
