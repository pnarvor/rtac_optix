#include <optix_helpers/Program.h>

namespace optix_helpers {

ProgramObj::ProgramObj(const Source& source, const Sources& headers,
                       const optix::Program& program) :
    source_(source),
    headers_(headers),
    program_(program)
{}

ProgramObj::ProgramObj(const ProgramObj& other) :
    source_(other.source_),
    headers_(other.headers_),
    program_(other.program_)
{}

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

optix::Handle<optix::VariableObj> ProgramObj::operator[](const std::string& varname)
{
    return program_[varname];
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

optix::Handle<optix::VariableObj> Program::operator[](const std::string& varname)
{
    return (*this)[varname];
}

RayGenerationProgramObj::RayGenerationProgramObj(const optix::Program& program,
                                                 const std::string& renderBufferName,
                                                 const Source& source, const Sources& headers) :
    ProgramObj(source, headers, program),
    renderBufferName_(renderBufferName)
{}

RayGenerationProgramObj::RayGenerationProgramObj(const ProgramObj& program,
                                                 const std::string& renderBufferName) :
    ProgramObj(program),
    renderBufferName_(renderBufferName)
{}

std::string RayGenerationProgramObj::render_buffer_name() const
{
    return renderBufferName_;
}

RayGenerationProgram::RayGenerationProgram() :
    Handle<RayGenerationProgramObj>()
{}

RayGenerationProgram::RayGenerationProgram(const optix::Program& program,
                                           const std::string& renderBufferName,
                                           const Source& source, const Sources& headers) :
    Handle<RayGenerationProgramObj>(new RayGenerationProgramObj(program, renderBufferName,
                                                                source, headers))
{}

RayGenerationProgram::RayGenerationProgram(const Program& program,
                                           const std::string& renderBufferName) :
    Handle<RayGenerationProgramObj>(new RayGenerationProgramObj(*program, renderBufferName))
{}

optix::Handle<optix::VariableObj> RayGenerationProgram::operator[](const std::string& varname)
{
    return (*this)[varname];
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


