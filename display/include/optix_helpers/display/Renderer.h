#ifndef _DEF_OPTIX_HELPERS_DISPLAY_RENDER_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_RENDER_H_

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>

#include <optix_helpers/display/utils.h>
#include <optix_helpers/display/View.h>

namespace optix_helpers { namespace display {

class RendererObj
{
    public:

    using Shape = ViewObj::Shape;

    static const Source vertexShader;
    static const Source fragmentShader;

    protected:
    
    GLuint renderProgram_;
    mutable View   view_;

    public:

    RendererObj(const Source& vertexShader = vertexShader,
                const Source& fragmentShader = fragmentShader,
                const View& view = View::New());
    
    virtual void draw();
    void set_view(const View& view) const;

    View view() const;
};

using Renderer = Handle<RendererObj>;

}; //namespace display
}; //namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_DISPLAY_RENDER_H_
