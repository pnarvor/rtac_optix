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

OptixProgramGroupOptions Pipeline::default_program_group_options()
{
    OptixProgramGroupOptions res;
    std::memset(&res, 0, sizeof(res)); // unused by optix for now
    return res;
}

Pipeline::Pipeline(const Context::ConstPtr& context) :
    context_(context),
    pipeline_(new OptixPipeline),
    compileOptions_(Pipeline::default_pipeline_compile_options()),
    linkOptions_   (Pipeline::default_pipeline_link_options())
{}

Pipeline::~Pipeline()
{
    // Destroying created programs.
    for(auto& program : programs_) {
        optixProgramGroupDestroy(program);
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
    OptixModule* module;

    // Checking if module already compiled.
    if(!forceReplace) {
        auto it = modules_.find(name);
        if(it != modules_.end()) {
            // A module with this name already exists. Ignoring compilation.
            return *it->second;
        }
    }
    module  = new OptixModule;
    *module = nullptr;

    OPTIX_CHECK( 
    optixModuleCreateFromPTX(*context_,
        &moduleOptions, &compileOptions_,
        ptxContent.c_str(), ptxContent.size(),
        nullptr, nullptr, // These are logging related, log will also
                          // be written in context log, but with less
                          // tracking information (TODO Fix this).
        module
        ) );

    // this allows to auto-delete the module once it is not referenced
    // anymore (the shared_ptr is keeping track of references).
    auto moduleDeleter = [](OptixModule* module) {
        if(*module != nullptr)
            OPTIX_CHECK( optixModuleDestroy(*module) );
        delete module;
    };

    modules_[name] = Handle<OptixModule>(module, moduleDeleter);
    return *module;
}

OptixModule Pipeline::module(const std::string& name)
{
    auto it = modules_.find(name);
    if(it == modules_.end())
        throw std::runtime_error("No module with name '" + name + "'");
    return *it->second;
}

OptixProgramGroup Pipeline::add_program_group(const OptixProgramGroupDesc& description)
{
    OptixProgramGroup program;
    auto opts = Pipeline::default_program_group_options();

    OPTIX_CHECK(
    optixProgramGroupCreate(*context_, &description, 1, &opts,
        nullptr, nullptr, // These are logging related, log will also
                          // be written in context log, but with less
                          // tracking information (TODO Fix this).
        &program));
    
    programs_.push_back(program);
    return program;
}

void Pipeline::link(bool autoStackSizes)
{
    OPTIX_CHECK(
    optixPipelineCreate(*context_, &compileOptions_, &linkOptions_,
        programs_.data(), programs_.size(), 
        nullptr, nullptr, // These are logging related, log will also
                          // be written in context log, but with less
                          // tracking information (TODO Fix this).
        &(*pipeline_)));
    
    if(autoStackSizes)
        this->autoset_stack_sizes();
}

void Pipeline::autoset_stack_sizes()
{
    // This is complicated...
    // Taken from the optix_triangle example in the OptiX-7.2 SDK
    OptixStackSizes stackSizes = {};
    for( auto& prog : programs_ ) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(prog, &stackSizes));
    }
    
    uint32_t directCallableStackSizeFromTraversal;
    uint32_t directCallableStackSizeFromState;
    uint32_t continuationStackSize;
    OPTIX_CHECK(
    optixUtilComputeStackSizes(&stackSizes, linkOptions_.maxTraceDepth,
        0,  // maxCCDepth ?
        0,  // maxDCDEpth ?
        &directCallableStackSizeFromTraversal,
        &directCallableStackSizeFromState,
        &continuationStackSize));
    OPTIX_CHECK(
    optixPipelineSetStackSize( (*pipeline_.get()),
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState, continuationStackSize,
        1  // maxTraversableDepth ?
        ) );

}

// BELOW HERE, ONLY METHOD OVERLOADS
OptixModule Pipeline::add_module(const std::string& name, const std::string& ptxContent,
                                 bool forceReplace)
{
    return this->add_module(name, ptxContent,
                            Pipeline::default_module_compile_options(),
                            forceReplace);
}

OptixProgramGroup Pipeline::add_raygen_program(const std::string& entryPoint,
                                               const std::string& moduleName)
{
    OptixProgramGroupDesc description    = {};
    description.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    description.raygen.module            = this->module(moduleName);
    description.raygen.entryFunctionName = entryPoint.c_str();

    return this->add_program_group(description);
}

OptixProgramGroup Pipeline::add_miss_program(const std::string& entryPoint,
                                             const std::string& moduleName)
{
    OptixProgramGroupDesc description    = {};
    description.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
    description.raygen.module            = this->module(moduleName);
    description.raygen.entryFunctionName = entryPoint.c_str();

    return this->add_program_group(description);
}

}; //namespace optix
}; //namespace rtac





