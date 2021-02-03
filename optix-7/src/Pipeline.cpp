#include <rtac_optix/Pipeline.h>

namespace rtac { namespace optix {

OptixPipelineCompileOptions Pipeline::default_pipeline_compile_options()
{
    OptixPipelineCompileOptions res;
    std::memset(&res, 0, sizeof(res));

    res.usesMotionBlur        = false;
    res.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    res.numPayloadValues      = 3;
    res.numAttributeValues    = 3;
    // compileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_DEBUG
    //                               | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH
    //                               | OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
    res.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    res.pipelineLaunchParamsVariableName = "params";
    res.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;

    return res;
}

OptixPipelineLinkOptions Pipeline::default_pipeline_link_options()
{
    OptixPipelineLinkOptions res;
    std::memset(&res, 0, sizeof(res));

    res.maxTraceDepth = 1;
    res.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    return res;
}

OptixModuleCompileOptions Pipeline::default_module_compile_options()
{
    OptixModuleCompileOptions res;
    std::memset(&res, 0, sizeof(res));

    res.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    res.optLevel         = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    res.debugLevel       = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

    return res;
}

Pipeline::Pipeline(const Context& context) :
    context_(context),
    pipeline_(new OptixPipeline),
    compileOptions_(Pipeline::default_pipeline_compile_options()),
    linkOptions_   (Pipeline::default_pipeline_link_options())
{}

Pipeline::~Pipeline()
{
    // Destroying created modules.
    for(auto& pair : modules_) {
        optixModuleDestroy(pair.second);
    }

    //Destroying pipeline
    if(*pipeline_) {
        optixPipelineDestroy(*pipeline_);
    }
}

Pipeline::operator OptixPipeline() const
{
    if(!(*pipeline_)) {
        throw std::runtime_error("Pipeline not initialized");
    }
    return *(pipeline_.get());
}

OptixPipelineCompileOptions Pipeline::compile_options() const
{
    return compileOptions_;
}

OptixPipelineLinkOptions Pipeline::link_options() const
{
    return linkOptions_;
}

OptixModule Pipeline::add_module(const std::string& name, const std::string& ptxContent,
                                 const OptixModuleCompileOptions& moduleOptions,
                                 bool forceReplace)
{
    OptixModule module = nullptr;

    // Checking if module already compiled.
    if(!forceReplace) {
        auto it = modules_.find(name);
        if(it != modules_.end()) {
            // A module with this name already exists. Ignoring compilation.
            return it->second;
        }
    }

    OPTIX_CHECK( optixModuleCreateFromPTX(
                context_,
                &moduleOptions,
                &compileOptions_,
                ptxContent.c_str(), ptxContent.size(),
                nullptr, nullptr, // These are logging related, log will also
                                  // be written in context log, but with less
                                  // tracking information (TODO Fix this).
                &module
                ) );

    modules_[name] = module;
    return module;
}

OptixModule Pipeline::module(const std::string& name)
{
    auto it = modules_.find(name);
    if(it == modules_.end())
        return nullptr;
    else
        return it->second;
}

OptixModule Pipeline::add_module(const std::string& name, const std::string& ptxContent,
                                 bool forceReplace)
{
    return this->add_module(name, ptxContent,
                            Pipeline::default_module_compile_options(),
                            forceReplace);
}

}; //namespace optix
}; //namespace rtac





