#ifndef _DEF_RTAC_OPTIX_PIPELINE_H_
#define _DEF_RTAC_OPTIX_PIPELINE_H_

#include <iostream>
#include <cstring>

#include <optix.h>

#include <rtac_optix/Handle.h>

namespace rtac { namespace optix {

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

}; //namespace optix
}; //namespace rtac

#endif //_DEF_RTAC_OPTIX_PIPELINE_H_
