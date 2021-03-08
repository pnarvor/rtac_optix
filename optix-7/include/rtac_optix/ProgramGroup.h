#ifndef _DEF_RTAC_PROGRAM_GROUP_H_
#define _DEF_RTAC_PROGRAM_GROUP_H_

#include <iostream>
#include <sstream>
#include <unordered_map>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/utils.h>
#include <rtac_optix/Handle.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/Module.h>
#include <rtac_optix/OptixWrapper.h>

namespace rtac { namespace optix {

class Pipeline;

class ProgramGroup : public OptixWrapper<OptixProgramGroup>
{
    public:

    friend class Pipeline;
    
    using Ptr      = OptixWrapperHandle<ProgramGroup>;
    using ConstPtr = OptixWrapperHandle<const ProgramGroup>;

    struct Function {

        static const char* Raygen;
        static const char* Miss;
        static const char* Exception;
        static const char* Intersection;
        static const char* AnyHit;
        static const char* ClosestHit;
        static const char* DirectCallable;
        static const char* ContinuationCallable;

        std::string name;
        Module::Ptr module;
    };
    struct FunctionNotFound : public std::runtime_error {
        FunctionNotFound(const std::string& kind) : 
            std::runtime_error(
                std::string("ProgramGroup has no \"") + kind + "\" function.")
        {}
    };

    using Kind        = OptixProgramGroupKind;
    using Description = OptixProgramGroupDesc;
    using Options     = OptixProgramGroupOptions;
    using Functions   = std::unordered_map<std::string, Function>;
    
    static Description empty_description(const Kind& kind,
        unsigned int flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE);

    // This is not used by optix-7. Leaving this here for future compatibility.
    static Options default_options();

    protected:
    
    Context::ConstPtr   context_;
    mutable Description description_;
    Options             options_;
    Functions           functions_;
    
    // Making these protected to force the user to use the set_* methods below
    Description& description();
    void add_function(const std::string& kind, const Function& function);

    void update_description() const;
    virtual void do_build() const;
    virtual void clean() const;

    ProgramGroup(const Context::ConstPtr& context, Kind kind,
                 unsigned int flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
                 const Options& options = default_options());
    
    static Ptr Create(const Context::ConstPtr& context, Kind kind,
                      unsigned int flags = OPTIX_PROGRAM_GROUP_FLAGS_NONE,
                      const Options& options = default_options());
    public:

    ~ProgramGroup();

    const Description& description() const;
    const Options& options() const;
    Options& options();

    Kind kind() const;
    unsigned int flags() const;
    void set_kind(Kind kind);

    Functions::const_iterator function(const std::string& kind) const;

    void set_raygen(const Function& function);
    void set_miss(const Function& function);
    void set_exception(const Function& function);
    void set_intersection(const Function& function);
    void set_anyhit(const Function& function);
    void set_closesthit(const Function& function);
    void set_direct_callable(const Function& function);
    void set_continuation_callable(const Function& function);
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_PROGRAM_GROUP_H_
