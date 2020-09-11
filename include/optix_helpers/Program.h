#ifndef _DEF_OPTIX_HELPERS_PROGRAM_H_
#define _DEF_OPTIX_HELPERS_PROGRAM_H_

#include <iostream>

#include <optixu/optixpp.h>

#include <optix_helpers/NamedObject.h>
#include <optix_helpers/Handle.h>
#include <optix_helpers/Source.h>
#include <optix_helpers/Buffer.h>

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

    void set_buffer(const Buffer& buffer);
    template <class NamedType>
    void set_object(const NamedType& object);

    optix::Handle<optix::VariableObj> operator[](const std::string& varname);
    optix::Handle<optix::VariableObj> operator[](const char* varname);

    const Source  source()      const;
    const Sources headers()     const;
    optix::Program program()    const; //? should be const ?
    operator optix::Program()   const;
    optix::Program operator->() const; //? should be const ?
};
using Program = Handle<ProgramObj>;

template <class NamedType>
void ProgramObj::set_object(const NamedType& object)
{
    program_[object->name()]->set(*object);
}

}; //namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Program& program);

#endif //_DEF_OPTIX_HELPERS_PROGRAM_H_
