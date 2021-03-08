#ifndef _DEF_RTAC_OPTIX_PIPELINE_H_
#define _DEF_RTAC_OPTIX_PIPELINE_H_

#include <iostream>
#include <cstring>
#include <unordered_map>
#include <vector>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>
#include <optix_stack_size.h> // This is to auto-compute the stack size (difficult)

#include <rtac_optix/utils.h>
#include <rtac_optix/Handle.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/ProgramGroup.h>

namespace rtac { namespace optix {

class Pipeline
{
    public:

    using Ptr      = Handle<Pipeline>;
    using ConstPtr = Handle<const Pipeline>;

    using Modules  = std::unordered_map<std::string, Module::Ptr>;
    using Programs = std::vector<ProgramGroup::Ptr>;
    
    using CompileOptions = OptixPipelineCompileOptions;
    using LinkOptions    = OptixPipelineLinkOptions;
    static CompileOptions default_pipeline_compile_options();
    static LinkOptions    default_pipeline_link_options();

    protected:

    mutable OptixPipeline pipeline_;
    Context::ConstPtr     context_;
    CompileOptions        compileOptions_;
    LinkOptions           linkOptions_;

    Modules  modules_;
    Programs programs_;

    void autoset_stack_sizes();
    virtual void do_build() const;
    virtual void destroy()  const;

    Pipeline(const Context::ConstPtr&           context,
             const CompileOptions& compileOptions,
             const LinkOptions&    linkOptions);

    public:

    static Ptr Create(const Context::ConstPtr& context,
        const CompileOptions& compileOptions = default_pipeline_compile_options(),
        const LinkOptions&    linkOptions    = default_pipeline_link_options());
    ~Pipeline();

    void link(bool autoStackSizes = true);
    // Implicitly castable to OptixPipeline for seamless use in optix API.
    // This breaks encapsulation.
    // /!\ Use only in optix API calls except for optixDeviceContextDestroy,
    operator OptixPipeline();

    const CompileOptions& compile_options() const;
    const LinkOptions&    link_options()    const;
    CompileOptions& compile_options();
    LinkOptions&    link_options();

    Module::Ptr add_module(const std::string& name, const std::string& ptxContent,
                           const Module::ModuleOptions& moduleOptions,
                           bool forceReplace = false);
    Module::Ptr      module(const std::string& name);
    Module::ConstPtr module(const std::string& name) const;

    ProgramGroup::Ptr add_program_group(ProgramGroup::Kind kind);

    // Starting from here, these are only overloads of already defined function
    // for convenience.
    Module::Ptr add_module(const std::string& name, const std::string& ptxContent,
                           bool forceReplace = false);

    ProgramGroup::Ptr add_raygen_program(const std::string& entryPoint,
                                         const std::string& moduleName);
    ProgramGroup::Ptr add_miss_program(const std::string& entryPoint,
                                       const std::string& moduleName);
    ProgramGroup::Ptr add_hit_programs();
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_PIPELINE_H_
