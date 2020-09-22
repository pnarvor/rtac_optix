#ifndef _DEF_OPTIX_HELPERS_RAY_GENERATOR_H_
#define _DEF_OPTIX_HELPERS_RAY_GENERATOR_H_

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/Buffer.h>

namespace optix_helpers {

template <typename ViewGeometryType, RTformat BufferFormat, class RenderBufferType>
class RayGeneratorObj
{
    public:

    struct BufferShape
    {
        size_t width;
        size_t height;
    };
    
    RenderBufferType renderBuffer_;
    Program          raygenProgram_;
    ViewGeometryType view_;

    public:
    
    RayGeneratorObj(const Context& context, const RayType& rayType,
                    const Source& raygenSource,
                    const std::string& renderBufferName = "renderBuffer");

    void set_size(size_t width, size_t height);

    ViewGeometryType view() const;
    BufferShape render_shape() const;
    RenderBufferType render_buffer() const;
};

template <typename ViewGeometryType, RTformat BufferFormat, class RenderBufferType>
using RayGenerator = Handle<RayGeneratorObj<ViewGeometryType, BufferFormat, RenderBufferType>>;

// implementation
template <typename ViewGeometryType, RTformat BufferFormat, class RenderBufferType>
RayGeneratorObj<ViewGeometryType, BufferFormat, RenderBufferType>::
RayGeneratorObj(const Context& context, const RayType& rayType,
                const Source& raygenSource, const std::string& renderBufferName) :
    renderBuffer_(context, BufferFormat, renderBufferName),
    raygenProgram_(context->create_program(raygenSource,
        {rayType->definition(), ViewGeometryType::rayGeometryDefinition})),
    view_(renderBuffer_, raygenProgram_)
{
    raygenProgram_->set_object(renderBuffer_);
}

template <typename ViewGeometryType, RTformat BufferFormat, class RenderBufferType>
void RayGeneratorObj<ViewGeometryType, BufferFormat, RenderBufferType>::
set_size(size_t width, size_t height)
{
    renderBuffer_->set_size(width, height);
    view_->set_size(width, height);
}

template <typename ViewGeometryType, RTformat BufferFormat, class RenderBufferType>
ViewGeometryType RayGeneratorObj<ViewGeometryType, BufferFormat, RenderBufferType>::view() const
{
    return view_;
}

template <typename ViewGeometryType, RTformat BufferFormat, class RenderBufferType>
typename RayGeneratorObj<ViewGeometryType, BufferFormat, RenderBufferType>::BufferShape
RayGeneratorObj<ViewGeometryType, BufferFormat, RenderBufferType>::render_shape() const
{
    BufferShape res;
    (*renderBuffer_)->getSize(res.width, res.height);
    return res;
}

template <typename ViewGeometryType, RTformat BufferFormat, class RenderBufferType>
RenderBufferType RayGeneratorObj<ViewGeometryType, BufferFormat, RenderBufferType>::render_buffer() const
{
    return renderBuffer_;
}

}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_RAY_GENERATOR_H_
