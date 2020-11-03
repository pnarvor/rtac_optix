#ifndef _DEF_OPTIX_HELPERS_DISPLAY_GL_BUFFER_H_
#define _DEF_OPTIX_HELPERS_DISPLAY_GL_BUFFER_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Handle.h>
#include <optix_helpers/Context.h>
#include <optix_helpers/Buffer.h>

#include <GL/glew.h>
//#define GL3_PROTOTYPES 1
#include <GL/gl.h>

namespace optix_helpers { namespace display {

class GLBuffer : public Buffer
{
    public:

    using Ptr      = Handle<GLBuffer>;
    using ConstPtr = Handle<const GLBuffer>;

    protected:
    
    GLuint       vboId_;
    RTbuffertype type_;

    static GLuint create_vbo();

    public:

    static Ptr New(const Context::ConstPtr& context,
                   RTbuffertype bufferType,
                   RTformat format,
                   const std::string& name);

    GLBuffer(const Context::ConstPtr& context,
             RTbuffertype bufferType,
             RTformat format,
             const std::string& name);

    virtual void set_size(size_t size);
    virtual void set_size(size_t width, size_t height);
    GLuint gl_id() const;
};

}; //namespace display
}; //namespace optix_helpers


#endif //_DEF_OPTIX_HELPERS_DISPLAY_GL_BUFFER_H_
