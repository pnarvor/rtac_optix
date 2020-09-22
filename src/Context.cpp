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

