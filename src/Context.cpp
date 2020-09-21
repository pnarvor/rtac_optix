#include <optix_helpers/Context.h>

namespace optix_helpers {

ContextObj::ContextObj(int entryPointCount) :
    context_(optix::Context::create())
{
    context_->setEntryPointCount(entryPointCount);
}

Program ContextObj::create_program(const Source& source, const Sources& additionalHeaders) const
{
    try {
        auto ptx = nvrtc_.compile(source, additionalHeaders);
        optix::Program program = context_->createProgramFromPTXString(ptx, source->name());
        return Program(new ProgramObj(source, additionalHeaders, program));
    }
    catch(const std::runtime_error& e) {
        std::ostringstream os;
        for(auto header : additionalHeaders) {
            os << header << "\n";
        }
        os << source <<  "\n" << e.what();
        throw std::runtime_error(os.str());
    }
}

Buffer ContextObj::create_buffer(RTbuffertype bufferType, RTformat format, 
                                 const std::string& name) const
{
    return Buffer(new BufferObj(context_->createBuffer(bufferType, format), name));
}

Buffer ContextObj::create_gl_buffer(RTbuffertype bufferType, RTformat format,
                                    unsigned int glboId, const std::string& name) const
{
    //return Buffer(new BufferObj(context_->createBufferFromGLBO(bufferType, format), name));
    auto res = Buffer(new BufferObj(context_->createBufferFromGLBO(bufferType, glboId), name));
    (*res)->setFormat(format);
    return res;
}

RayType ContextObj::create_raytype(const Source& rayDefinition) const
{
    unsigned int rayTypeIndex = context_->getRayTypeCount();
    context_->setRayTypeCount(rayTypeIndex + 1);
    return RayType(rayTypeIndex, rayDefinition);
}

optix::Handle<optix::VariableObj> ContextObj::operator[](const std::string& varname)
{
    return context_[varname];
}

ContextObj::operator optix::Context() const
{
    return context_;
}

optix::Context ContextObj::operator->()
{
    return context_;
}

optix::Context ContextObj::operator->() const
{
    return context_;
}

optix::Context ContextObj::context() const
{
    return context_;
}

Context::Context(int entryPointCount) :
    Handle<ContextObj>(new ContextObj(entryPointCount))
{}

} //namespace optix_helpers

