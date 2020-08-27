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
    optix::Program program_;

    public:

    ProgramObj(const Source& source, const Sources& headers,
               const optix::Program& program);

    const Source  source()  const;
    const Sources headers() const;
    optix::Program program(); //? should be const ?
};

class Program : public Handle<ProgramObj>
{
    public:

    Program();
    Program(const Source& source, const Sources& headers,
            const optix::Program& program);
};

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Program& program);

#endif //_DEF_OPTIX_HELPERS_PROGRAM_H_
