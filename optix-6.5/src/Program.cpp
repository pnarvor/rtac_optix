#include <optix_helpers/Program.h>

namespace optix_helpers {

Program::Ptr Program::New(const Source::ConstPtr& source, const Sources& headers,
                          const optix::Program& program)
{
    return Ptr(new Program(source, headers, program));
}

Program::Program(const Source::ConstPtr& source, const Sources& headers,
                 const optix::Program& program) :
    source_(source),
    headers_(headers),
    program_(program)
{}

Source::ConstPtr Program::source() const
{
    return source_;
}

const Sources Program::headers() const
{
    return headers_;
}

optix::Handle<optix::VariableObj> Program::operator[](const std::string& varname)
{
    return program_[varname];
}

optix::Handle<optix::VariableObj> Program::operator[](const char* varname)
{
    return program_[varname];
}

optix::Program Program::program() const
{
    return program_;
}

Program::operator optix::Program() const
{
    return program_;
}

optix::Program Program::operator->() const
{
    return program_;
}

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Program& program)
{
    os << "Program " << program.source()->name() << " :\n";
    for(auto header : program.headers()) {
        os << header << "\n";
    }
    os << program.source() << "\n";

    return os;
}

std::ostream& operator<<(std::ostream& os, const optix_helpers::Program::ConstPtr& program)
{
    os << *program;
    return os;
}

std::ostream& operator<<(std::ostream& os, const optix_helpers::Program::Ptr& program)
{
    os << *program;
    return os;
}

