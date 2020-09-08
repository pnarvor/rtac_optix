#ifndef _DEF_OPTIX_HELPERS_PROGRAM_H_
#define _DEF_OPTIX_HELPERS_PROGRAM_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/Source.h>

namespace optix_helpers {

class Program
{
    protected:

    Source         source_;
    Sources        headers_;
    mutable optix::Program program_; // fix this

    public:

    Program(const Source& source, const Sources& headers,
               const optix::Program& program);
    Program(const Program& other);

    const Source  source()   const;
    const Sources headers()  const;
    optix::Program program() const; //? should be const ?

    optix::Handle<optix::VariableObj> operator[](const std::string& varname);
    optix::Handle<optix::VariableObj> operator[](const char* varname);

    operator optix::Program() const;
};

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Program& program);

#endif //_DEF_OPTIX_HELPERS_PROGRAM_H_
