#include <optix_helpers/Program.h>

namespace optix_helpers {

ProgramObj::ProgramObj(const Source& source, const Sources& headers,
                       const optix::Program& program) :
    source_(source),
    headers_(headers),
    program_(program)
{
}

const Source ProgramObj::source() const
{
    return source_;
}

const Sources ProgramObj::headers() const
{
    return headers_;
}

optix::Program ProgramObj::program() const
{
    return program_;
}

Program::Program() :
    Handle<ProgramObj>()
{}

Program::Program(const Source& source, const Sources& headers,
                 const optix::Program& program) :
    Handle<ProgramObj>(new ProgramObj(source, headers, program))
{
}

Program::operator optix::Program() const
{
    return (*this)->program();
}

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Program& program)
{
    if(!program) {
        os << "Empty program.\n";
        return os;
    }
    os << "Program " << program->source()->name() << " :\n";
    for(auto header : program->headers()) {
        os << header << "\n";
    }
    os << program->source() << "\n";

    return os;
}


