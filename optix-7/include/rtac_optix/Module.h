#ifndef _DEF_RTAC_OPTIX_MODULE_H_
#define _DEF_RTAC_OPTIX_MODULE_H_

#include <iostream>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>
#include <rtac_optix/Context.h>

namespace rtac { namespace optix {

// Formward declaration of class Pipeline (only pipeline will be allowed to
// Create a new Module
class Pipeline;

class Module
{
    public:

    friend class Pipeline;

    using Ptr      = Handle<Module>;
    using ConstPtr = Handle<const Module>;

    using PipelineOptions = OptixPipelineCompileOptions;
    using ModuleOptions   = OptixModuleCompileOptions;
    static ModuleOptions default_options();

    protected:
    
    mutable OptixModule module_;
    Context::ConstPtr   context_;
    std::string         ptxSource_;
    PipelineOptions     pipelineOptions_;
    ModuleOptions       moduleOptions_;

    virtual void do_build() const;
    virtual void destroy() const;

    Module(const Context::ConstPtr& context,
           const std::string& ptxSource,
           const PipelineOptions& pipelineOptions,
           const ModuleOptions& moduleOptions);

    static Ptr Create(const Context::ConstPtr& context,
                      const std::string& ptxSource,
                      const PipelineOptions& pipelineOptions,
                      const ModuleOptions& options = default_options());

    public:

    ~Module();

    const PipelineOptions& pipeline_options() const;
    const ModuleOptions&   module_options() const;
    // These getters will invalidate the pipeline which will automatically be
    // rebuilt on cast to OptixPipeline type.
    PipelineOptions& pipeline_options();
    ModuleOptions&   module_options();

    operator OptixModule() const; // this will trigger build
};

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_MODULE_H_
