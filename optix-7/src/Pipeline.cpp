#include <rtac_optix/Pipeline.h>

namespace rtac { namespace optix {

Pipeline::CompileOptions Pipeline::default_pipeline_compile_options()
{
    auto res = zero<CompileOptions>();

    res.usesMotionBlur        = false;
    //res.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    res.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
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

Pipeline::LinkOptions Pipeline::default_pipeline_link_options()
{
    auto res = zero<LinkOptions>();

    res.maxTraceDepth = 1;
    res.debugLevel    = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    return res;
}

Pipeline::Pipeline(const Context::ConstPtr& context,
                   const CompileOptions& compileOptions,
                   const LinkOptions& linkOptions) :
    context_(context),
    compileOptions_(compileOptions),
    linkOptions_(linkOptions),
    compileOptionsChanged_(false)
{}

Pipeline::Ptr Pipeline::Create(const Context::ConstPtr& context,
                               const CompileOptions& compileOptions,
                               const LinkOptions& linkOptions)
{
    return Ptr(new Pipeline(context, compileOptions, linkOptions));
}

Pipeline::~Pipeline()
{
    try {
        this->clean();
    }
    catch(const std::runtime_error& e) {
        std::cerr << "Caught exception during rtac::optix::Pipeline destruction : " 
                  << e.what() << std::endl;
    }
}

void Pipeline::build() const
{
    if(compileOptionsChanged_) {
        for(auto module : modules_) {
            module.second->pipeline_options() = compileOptions_;
        }
        compileOptionsChanged_ = false;
    }
    this->OptixWrapper<OptixPipeline>::build();
}

void Pipeline::do_build() const
{
    std::vector<OptixProgramGroup> compiledPrograms(programs_.size());
    for(int i = 0; i < programs_.size(); i++) {
        compiledPrograms[i] = *programs_[i];
    }

    OPTIX_CHECK(
    optixPipelineCreate(*context_, &compileOptions_, &linkOptions_,
        compiledPrograms.data(), compiledPrograms.size(), 
        nullptr, nullptr, // These are logging related, log will also
                          // be written in context log, but with less
                          // tracking information (TODO Fix this).
        &optixObject_));
}

void Pipeline::clean() const
{
    if(optixObject_) {
        OPTIX_CHECK( optixPipelineDestroy(optixObject_) );
    }
}

void Pipeline::autoset_stack_sizes()
{
    // Not used anymore

    // From the docs : defaults values will be set if none are given (if
    // maximum depth of call trees CC and DC programs is at most 2)

    // This is complicated... See the Optix 7 documentation.
    // Taken from the optix_triangle example in the OptiX-7.2 SDK
    OptixStackSizes stackSizes = {};
    for( auto& prog : programs_ ) {
        OPTIX_CHECK(optixUtilAccumulateStackSizes(*prog, &stackSizes));
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
    optixPipelineSetStackSize( optixObject_,
        directCallableStackSizeFromTraversal,
        directCallableStackSizeFromState, continuationStackSize,
        1  // maxTraversableDepth ?
        ) );

}

const Pipeline::CompileOptions& Pipeline::compile_options() const
{
    return compileOptions_;
}

const Pipeline::LinkOptions& Pipeline::link_options() const
{
    return linkOptions_;
}

Pipeline::CompileOptions& Pipeline::compile_options()
{
    this->bump_version();
    compileOptionsChanged_ = true;
    return compileOptions_;
}

Pipeline::LinkOptions& Pipeline::link_options()
{
    this->bump_version();
    return linkOptions_;
}

Module::Ptr Pipeline::add_module(const std::string& name, const std::string& ptxContent,
                                 const Module::ModuleOptions& moduleOptions,
                                 bool forceReplace)
{
    // Checking if module already compiled.
    if(!forceReplace) {
        auto it = modules_.find(name);
        if(it != modules_.end()) {
            return it->second;
        }
    }
    
    this->bump_version();
    auto module = Module::Create(context_, ptxContent,
                                 compileOptions_, moduleOptions);
    modules_[name] = module;
    this->add_dependency(module);
    return module;
}

Module::Ptr Pipeline::module(const std::string& name)
{
    auto it = modules_.find(name);
    if(it == modules_.end())
        throw std::runtime_error("No module with name '" + name + "'");
    return it->second;
}

Module::ConstPtr Pipeline::module(const std::string& name) const
{
    auto it = modules_.find(name);
    if(it == modules_.end())
        throw std::runtime_error("No module with name '" + name + "'");
    return it->second;
}
bool Pipeline::contains_module(const Module::ConstPtr& module) const
{
    for(auto& item : modules_) {
        if(item.second == module)
            return true;
    }
    return false;
}

ProgramGroup::Ptr Pipeline::add_program_group(ProgramGroup::Kind kind)
{
    auto program = ProgramGroup::Create(context_, kind);
    programs_.push_back(program);
    this->add_dependency(program);
    return program;
}

// BELOW HERE, ONLY METHOD OVERLOADS
Module::Ptr Pipeline::add_module(const std::string& name, const std::string& ptxContent,
                                 bool forceReplace)
{
    return this->add_module(name, ptxContent,
                            Module::default_options(), forceReplace);
}

ProgramGroup::Ptr Pipeline::add_raygen_program(const std::string& entryPoint,
                                               const std::string& moduleName)
{
    auto program = this->add_program_group(OPTIX_PROGRAM_GROUP_KIND_RAYGEN);
    program->set_raygen({entryPoint, this->module(moduleName)});
    return program;
}

ProgramGroup::Ptr Pipeline::add_raygen_program(const std::string& entryPoint,
                                               const Module::Ptr& module)
{
    if(!this->contains_module(module)) {
        throw std::runtime_error("Trying to create a program with a foreign module");
    }
    auto program = this->add_program_group(OPTIX_PROGRAM_GROUP_KIND_RAYGEN);
    program->set_raygen({entryPoint, module});
    return program;
}

ProgramGroup::Ptr Pipeline::add_miss_program(const std::string& entryPoint,
                                             const std::string& moduleName)
{
    auto program = this->add_program_group(OPTIX_PROGRAM_GROUP_KIND_MISS);
    program->set_miss({entryPoint, this->module(moduleName)});
    return program;
}

ProgramGroup::Ptr Pipeline::add_miss_program(const std::string& entryPoint,
                                             const Module::Ptr& module)
{
    if(!this->contains_module(module)) {
        throw std::runtime_error("Trying to create a program with a foreign module");
    }
    auto program = this->add_program_group(OPTIX_PROGRAM_GROUP_KIND_MISS);
    program->set_miss({entryPoint, module});
    return program;
}

ProgramGroup::Ptr Pipeline::add_hit_programs()
{
    return this->add_program_group(OPTIX_PROGRAM_GROUP_KIND_HITGROUP);
}

}; //namespace optix
}; //namespace rtac





