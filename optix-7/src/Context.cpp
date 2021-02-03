#include <rtac_optix/Context.h>

namespace rtac { namespace optix {

Context::ContextPtr Context::new_context(const OptixDeviceContextOptions& options,
                                         CUcontext cudaContext)
{
    auto context = new OptixDeviceContext;
    
    OPTIX_CHECK( optixDeviceContextCreate(cudaContext, &options, context) );
    
    // this allows to auto-delete the context once it is not referenced
    // anymore (the shared_ptr is keeping track of references).
    auto contextDeleter = [](OptixDeviceContext* ctx) {
        OPTIX_CHECK( optixDeviceContextDestroy(*ctx) );
        delete ctx;
    };
    return ContextPtr(context, contextDeleter);
}

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

Context::Context() :
    options_(Context::default_options()),
    context_(Context::new_context(options_))
{
}

Context::operator OptixDeviceContext() const
{
    return *(context_.get());
}

}; //namespace optix
}; //namespace rtac

