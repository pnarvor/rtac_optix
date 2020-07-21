#ifndef _DEF_OPTIX_HELPERS_PROGRAM_MANAGER_H_
#define _DEF_OPTIX_HELPERS_PROGRAM_MANAGER_H_

#include <iostream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <memory>
#include <unordered_map>
#include <list>

#include <optixu/optixpp.h>

#include <optix_helpers/NVRTC_Helper.h>

namespace optix_helpers {


using StringPtr = std::shared_ptr<const std::string>;

class Program
{
    protected:
    
    StringPtr      cuString_;
    StringPtr      ptxString_;
    std::string    name_;
    optix::Program program_;
    
    public:

    Program();
    Program(const StringPtr& cuString, const StringPtr& ptxString,
            const std::string& functionName, const optix::Program& program);
    Program(const Program& other);

    StringPtr      cu_string() const;
    StringPtr      ptx_string() const;
    std::string    name() const;
    optix::Program program() const;
    bool operator!() const;

    //static functions:
    static void print_source(const std::string& source, std::ostream& os);
    static std::string print_source(const std::string& source);
};

class Context;
using ContextPtr      = std::shared_ptr<Context>;
using ContextConstPtr = std::shared_ptr<const Context>;

class Context
{
    //The point of this class is to manage cached compiled files.
    //None of this is implemented yet. For now is only a "dumb" compiler.
    protected:

    optix::Context context_;
    Nvrtc nvrtc_;

    public:

    static ContextPtr create();

    Context();
    Context(const Context& other);
    
    Program from_cufile(const std::string& path,
                        const std::string& functionName);
    Program from_custring(const std::string& cuString,
                          const std::string& functionName);

    optix::Context context();
};

}; // namespace optix_helpers

std::ostream& operator<<(std::ostream& os, const optix_helpers::Program& program);


#endif //_DEF_OPTIX_HELPERS_PROGRAM_MANAGER_H_
