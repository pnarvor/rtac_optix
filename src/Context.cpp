#include <optix_helpers/Context.h>

namespace optix_helpers {

Context::Ptr Context::New(int entryPointCount)
{
    return Ptr(new Context(entryPointCount));
}

Context::Context(int entryPointCount) :
    context_(optix::Context::create())
{
    context_->setEntryPointCount(entryPointCount);
}

Program::Ptr Context::create_program(const Source::ConstPtr& source,
                                     const Sources& additionalHeaders) const
{
    try {
        auto ptx = nvrtc_.compile(source, additionalHeaders);
        optix::Program program = context_->createProgramFromPTXString(ptx, source->name());
        return Program::New(source, additionalHeaders, program);
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

optix::Handle<optix::VariableObj> Context::operator[](const std::string& varname)
{
    return context_[varname];
}

Context::operator optix::Context() const
{
    return context_;
}

optix::Context Context::operator->()
{
    return context_;
}

optix::Context Context::operator->() const
{
    return context_;
}

optix::Context Context::context() const
{
    return context_;
}

} //namespace optix_helpers


