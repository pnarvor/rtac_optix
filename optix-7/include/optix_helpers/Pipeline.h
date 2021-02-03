#ifndef _DEF_OPTIX_HELPERS_PIPELINE_H_
#define _DEF_OPTIX_HELPERS_PIPELINE_H_

#include <iostream>
#include <cstring>

#include <rtac_base/types/Handle.h>

#include <optix.h>

namespace optix_helpers {

class Pipeline
{
    protected:

    OptixPipelineCompileOptions compileOptions_;
    OptixPipelineLinkOptions    linkOptions_;
    OptixModuleCompileOptions   moduleOptions_;

    OptixPipeline pipeline_;

    public:

    static OptixPipelineCompileOptions default_pipeline_compile_options();
    static OptixPipelineLinkOptions    default_pipeline_link_options();
    static OptixModuleCompileOptions   default_module_compile_options();

    Pipeline();
};

}; // namespace optix_helpers

#endif //_DEF_OPTIX_HELPERS_PIPELINE_H_
