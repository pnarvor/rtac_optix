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

namespace rtac { namespace optix {

class Pipeline
{
    public:

    using PipelinePtr = Handle<OptixPipeline>;
    using ModuleDict  = std::unordered_map<std::string, Handle<OptixModule>>;
    using Programs    = std::vector<OptixProgramGroup>;

    static OptixPipelineCompileOptions default_pipeline_compile_options();
    static OptixPipelineLinkOptions    default_pipeline_link_options();
    static OptixModuleCompileOptions   default_module_compile_options();
    static OptixProgramGroupOptions    default_program_group_options();

    protected:

    Context             context_;
    mutable PipelinePtr pipeline_;
    ModuleDict          modules_;
    Programs            programs_;

    OptixPipelineCompileOptions compileOptions_;
    OptixPipelineLinkOptions    linkOptions_;

    void autoset_stack_sizes();

    public:

    Pipeline(const Context& context);
    ~Pipeline();
    operator OptixPipeline() const;


    OptixPipelineCompileOptions compile_options() const;
    OptixPipelineLinkOptions    link_options()    const;

    OptixModule add_module(const std::string& name, const std::string& ptxContent,
                           const OptixModuleCompileOptions& moduleOptions,
                           bool forceReplace = false);
    OptixModule module(const std::string& name);

    OptixProgramGroup add_program_group(const OptixProgramGroupDesc& description);

    void link(bool autoStackSizes = true);

    // Starting from here, these are only overloads of already defined function
    // for convenience.
    OptixModule add_module(const std::string& name, const std::string& ptxContent,
                           bool forceReplace = false);

    OptixProgramGroup add_raygen_program(const std::string& entryPoint,
                                         const std::string& moduleName);
    OptixProgramGroup add_miss_program(const std::string& entryPoint,
                                       const std::string& moduleName);
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_PIPELINE_H_
