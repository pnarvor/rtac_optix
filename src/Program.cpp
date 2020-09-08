#include <optix_helpers/Program.h>

namespace optix_helpers {

Program::Program(const Source& source, const Sources& headers,
                       const optix::Program& program) :
    source_(source),
    headers_(headers),
    program_(program)
{}

Program::Program(const Program& other) :
    source_(other.source_),
    headers_(other.headers_),
    program_(other.program_)
{}

const Source Program::source() const
{
    return source_;
}

const Sources Program::headers() const
{
    return headers_;
}

optix::Program Program::program() const
{
    return program_;
}

optix::Handle<optix::VariableObj> Program::operator[](const std::string& varname)
{
    return program_[varname];
}

optix::Handle<optix::VariableObj> Program::operator[](const char* varname)
{
    return program_[varname];
}

Program::operator optix::Program() const
{
    return program_;
}

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Program& program)
{
    os << "Program " << program.source().name() << " :\n";
    for(auto header : program.headers()) {
        os << header << "\n";
    }
    os << program.source() << "\n";

    return os;
}


