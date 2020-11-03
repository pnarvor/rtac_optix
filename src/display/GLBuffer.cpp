#include <optix_helpers/display/GLBuffer.h>

namespace optix_helpers { namespace display {

GLBuffer::Ptr GLBuffer::New(const Context::ConstPtr& context,
                            RTbuffertype bufferType,
                            RTformat format,
                            const std::string& name)
{
    return Ptr(new GLBuffer(context, bufferType, format, name));
}

GLBuffer::GLBuffer(const Context::ConstPtr& context,
                   RTbuffertype bufferType,
                   RTformat format,
                   const std::string& name) :
    Buffer((*context)->createBufferFromGLBO(
           bufferType, GLBuffer::create_vbo()),
           name),
    vboId_(this->buffer()->getGLBOId())
{
    this->buffer()->setFormat(format);
}

GLuint GLBuffer::create_vbo()
{
    GLuint vboId;
    glGenBuffers(1, &vboId);
    return vboId;
}

void GLBuffer::set_size(size_t size)
{
    this->Buffer::set_size(size);
    size_t bsize = size*this->buffer()->getElementSize();
    glBindBuffer(GL_ARRAY_BUFFER, vboId_);
    glBufferData(GL_ARRAY_BUFFER, bsize, NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLBuffer::set_size(size_t width, size_t height)
{
    this->Buffer::set_size(width, height);
    size_t bsize = width*height*this->buffer()->getElementSize();
    glBindBuffer(GL_ARRAY_BUFFER, vboId_);
    glBufferData(GL_ARRAY_BUFFER, bsize, NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

GLuint GLBuffer::gl_id() const
{
    return vboId_;
}

}; //namespace display
}; //namespace optix_helpers

