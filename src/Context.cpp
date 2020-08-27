#include <optix_helpers/Context.h>

namespace optix_helpers {

ContextObj::ContextObj() :
    context_(optix::Context::create())
{
}


Program ContextObj::create_program(const Source& source, const Sources& additionalHeaders) const
{
    try {
        auto ptx = nvrtc_.compile(source, additionalHeaders);
        optix::Program program = context_->createProgramFromPTXString(ptx, source->name());
        return Program(source, additionalHeaders, program);
    }
    catch(const std::runtime_error& e) {
        std::ostringstream os;
        os << source <<  "\n" << e.what();
        throw std::runtime_error(os.str());
    }
}

optix::Context ContextObj::context() const
{
    return context_;
}

Context::Context() :
    Handle<ContextObj>(new ContextObj)
{}

} //namespace optix_helpers

