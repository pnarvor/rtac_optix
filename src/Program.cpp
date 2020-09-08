#include <optix_helpers/Program.h>

namespace optix_helpers {

ProgramObj::ProgramObj(const Source& source, const Sources& headers,
                       const optix::Program& program) :
    source_(source),
    headers_(headers),
    program_(program)
{}

void ProgramObj::set_buffer(const Buffer& buffer)
{
    program_[buffer->name()]->set(*buffer);
}

const Source ProgramObj::source() const
{
    return source_;
}

const Sources ProgramObj::headers() const
{
    return headers_;
}

optix::Handle<optix::VariableObj> ProgramObj::operator[](const std::string& varname)
{
    return program_[varname];
}

optix::Handle<optix::VariableObj> ProgramObj::operator[](const char* varname)
{
    return program_[varname];
}

optix::Program ProgramObj::program() const
{
    return program_;
}

ProgramObj::operator optix::Program() const
{
    return program_;
}

optix::Program ProgramObj::operator->() const
{
    return program_;
}

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Program& program)
{
    os << "Program " << program->source()->name() << " :\n";
    for(auto header : program->headers()) {
        os << header << "\n";
    }
    os << program->source() << "\n";

    return os;
}


