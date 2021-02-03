#include <optix_helpers/Pipeline.h>

namespace optix_helpers {

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

Pipeline::Pipeline() :
    compileOptions_(Pipeline::default_pipeline_compile_options()),
    linkOptions_   (Pipeline::default_pipeline_link_options()),
    moduleOptions_ (Pipeline::default_module_compile_options()),
    pipeline_(nullptr)
{}

}; // namespace optix_helpers




