#ifndef _DEF_OPTIX_HELPERS_PROGRAM_H_
#define _DEF_OPTIX_HELPERS_PROGRAM_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Source.h>

namespace optix_helpers {

class ProgramObj
{
    protected:

    Source         source_;
    Sources        headers_;
    mutable optix::Program program_; // fix this

    public:

    ProgramObj(const Source& source, const Sources& headers,
               const optix::Program& program);
    ProgramObj(const ProgramObj& other);

    const Source  source()   const;
    const Sources headers()  const;
    optix::Program program() const; //? should be const ?

    optix::Handle<optix::VariableObj> operator[](const std::string& varname);
};

class Program : public Handle<ProgramObj>
{
    public:

    Program();
    Program(const Source& source, const Sources& headers,
            const optix::Program& program);

    operator optix::Program() const; // implicit conversion to optix native type
    optix::Handle<optix::VariableObj> operator[](const std::string& varname);
};

class RayGenerationProgramObj : public ProgramObj
{
    protected:

    std::string renderBufferName_;

    public:

    RayGenerationProgramObj(const optix::Program& program,
                            const std::string& renderBufferName,
                            const Source& source, const Sources& headers = Sources());
    RayGenerationProgramObj(const ProgramObj& program, const std::string& renderBufferName);

    std::string render_buffer_name() const;
};

class RayGenerationProgram : public Handle<RayGenerationProgramObj>
{
    public:

    RayGenerationProgram();
    RayGenerationProgram(const optix::Program& program,
                         const std::string& renderBufferName,
                         const Source& source, const Sources& headers = Sources());
    RayGenerationProgram(const Program& program, const std::string& renderBufferName);

    operator optix::Program() const; // implicit conversion to optix native type
    optix::Handle<optix::VariableObj> operator[](const std::string& varname);
};

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Program& program);

#endif //_DEF_OPTIX_HELPERS_PROGRAM_H_
