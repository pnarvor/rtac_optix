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

    using Ptr      = Handle<Pipeline>;
    using ConstPtr = Handle<const Pipeline>;

    using ModuleDict  = std::unordered_map<std::string, OptixModule>;
    using Programs    = std::vector<OptixProgramGroup>;

    static OptixPipelineCompileOptions default_pipeline_compile_options();
    static OptixPipelineLinkOptions    default_pipeline_link_options();
    static OptixModuleCompileOptions   default_module_compile_options();
    static OptixProgramGroupOptions    default_program_group_options();

    protected:

    Context::ConstPtr     context_;
    mutable OptixPipeline pipeline_;
    ModuleDict            modules_;
    Programs              programs_;

    OptixPipelineCompileOptions compileOptions_;
    OptixPipelineLinkOptions    linkOptions_;

    void autoset_stack_sizes();

    Pipeline(const Context::ConstPtr&           context,
             const OptixPipelineCompileOptions& compileOptions,
             const OptixPipelineLinkOptions&    linkOptions);

    public:

    static Ptr Create(const Context::ConstPtr& context,
        const OptixPipelineCompileOptions& compileOptions = default_pipeline_compile_options(),
        const OptixPipelineLinkOptions&    linkOptions    = default_pipeline_link_options());
    ~Pipeline();

    // Implicitly castable to OptixPipeline for seamless use in optix API.
    // This breaks encapsulation.
    // /!\ Use only in optix API calls except for optixDeviceContextDestroy,
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
