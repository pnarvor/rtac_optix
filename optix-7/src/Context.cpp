#include <rtac_optix/Context.h>
#include <optix_function_table_definition.h> // ?

namespace rtac { namespace optix {

OptixDeviceContextOptions Context::default_options()
{
    OptixDeviceContextOptions res;
    std::memset(&res, 0, sizeof(res));

    res.logCallbackFunction = &Context::log_callback;
    res.logCallbackLevel    = 4;

    return res;
}

void Context::log_callback( unsigned int level, const char* tag, const char* message, void* data)
{
    std::cerr << "[" << std::setw( 2 ) << level << "][" << std::setw( 12 ) << tag << "]: "
              << message << "\n";
}

Context::Context(const OptixDeviceContext& context) :
    context_(context)
{}

Context::Ptr Context::Create(const OptixDeviceContextOptions& options,
                             CUcontext cudaContext,
                             bool diskCacheEnabled)
{
    optix_init();
    OptixDeviceContext context;
    OPTIX_CHECK( optixDeviceContextCreate(cudaContext, &options, &context) );

    auto res = Ptr(new Context(context));
    if(diskCacheEnabled)
        res->enable_cache();
    else
        res->disable_cache();
    return res;
}

Context::~Context()
{
    try {
        OPTIX_CHECK( optixDeviceContextDestroy(context_) );
    }
    catch(const std::runtime_error& e) {
        std::cerr << "Caught exception during rtac::optix::Context destruction : " 
                  << e.what() << std::endl;
    }
}

Context::operator OptixDeviceContext() const
{
    return context_;
}

void Context::enable_cache()
{
    OPTIX_CHECK( optixDeviceContextSetCacheEnabled(context_, 1) );
}

void Context::disable_cache()
{
    OPTIX_CHECK( optixDeviceContextSetCacheEnabled(context_, 0) );
}

}; //namespace optix
}; //namespace rtac

