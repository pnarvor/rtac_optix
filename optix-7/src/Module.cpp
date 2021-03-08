#include <rtac_optix/Module.h>

namespace rtac { namespace optix {

Module::ModuleOptions Module::default_options()
{
    auto options = zero<ModuleOptions>();

    options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    options.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    options.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    return options;
}

Module::Module(const Context::ConstPtr& context,
               const std::string& ptxSource,
               const Module::PipelineOptions& pipelineOptions,
               const Module::ModuleOptions& moduleOptions) :
    context_(context),
    ptxSource_(ptxSource),
    pipelineOptions_(pipelineOptions),
    moduleOptions_(moduleOptions)
{}

Module::Ptr Module::Create(const Context::ConstPtr& context,
                           const std::string& ptxSource,
                           const Module::PipelineOptions& pipelineOptions,
                           const Module::ModuleOptions& moduleOptions)
{
    return Ptr(new Module(context, ptxSource, pipelineOptions, moduleOptions));
}

Module::~Module()
{
    try {
        this->clean();
    }
    catch(const std::exception& e) {
        std::cerr << "Caught exception during rtac::optix::Module destruction : " 
                  << e.what() << std::endl;
    }
}

void Module::do_build() const
{
    OPTIX_CHECK( 
    optixModuleCreateFromPTX(*context_,
        &moduleOptions_, &pipelineOptions_,
        ptxSource_.c_str(), ptxSource_.size(),
        nullptr, nullptr, // These are logging related, log will also be
                          // written in context log, but with less tracking
                          // information (TODO Fix this. See a unified
                          // interface with Context type ?)
        &optixObject_
        ) );
}

void Module::clean() const
{
    if(optixObject_)
        OPTIX_CHECK( optixModuleDestroy(optixObject_) );
}

const Module::PipelineOptions& Module::pipeline_options() const
{
    return pipelineOptions_;
}

const Module::ModuleOptions& Module::module_options() const
{
    return moduleOptions_;
}

Module::PipelineOptions& Module::pipeline_options()
{
    this->bump_version();
    return pipelineOptions_;
}

Module::ModuleOptions& Module::module_options()
{
    this->bump_version();
    return moduleOptions_;
}

}; //namespace optix
}; //namespace rtac
