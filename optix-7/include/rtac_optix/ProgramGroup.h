#ifndef _DEF_RTAC_PROGRAM_GROUP_H_
#define _DEF_RTAC_PROGRAM_GROUP_H_

#include <iostream>
#include <sstream>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/utils.h>
#include <rtac_optix/Handle.h>
#include <rtac_optix/Context.h>

namespace rtac { namespace optix {

class ProgramGroup
{
    public:

    using Ptr      = Handle<ProgramGroup>;
    using ConstPtr = Handle<const ProgramGroup>;
    
    // This is not used by optix-7. Leaving this here for future compatibility.
    static OptixProgramGroupOptions default_options();

    protected:
    
    Context::ConstPtr         context_;
    mutable OptixProgramGroup program_;
    OptixProgramGroupDesc     description_;
    OptixProgramGroupOptions  options_;
    
    // The entry function names are saved in these strings. The
    // OptixProgramGroupDesc only saves const char* to the function names and
    // that migth be an issue if the original string which were used to fill
    // the description goes out of scope.
    std::string entryFunctionNames_[3];

    ProgramGroup(const Context::ConstPtr&        context,
                 const OptixProgramGroupDesc&    description,
                 const OptixProgramGroupOptions& options = default_options());
    
    private:

    void store_entry_function_names();
    void store_entry_function_name(std::string& dst, const char** src);

    public:

    static Ptr Create(const Context::ConstPtr&        context,
                      const OptixProgramGroupDesc&    description,
                      const OptixProgramGroupOptions& options = default_options());
    ~ProgramGroup();

    OptixProgramGroup build();

    operator OptixProgramGroup();
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_PROGRAM_GROUP_H_
