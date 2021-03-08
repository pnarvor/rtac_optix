#ifndef _RTAC_OPTIX_TRAVERSABLE_HANDLE_H_
#define _RTAC_OPTIX_TRAVERSABLE_HANDLE_H_

#include <iostream>
#include <iomanip>
#include <cstring>

#include <optix.h>
// careful : because of OptiX function table optix_stubs.h must be included to
// ensure proper linking.
#include <optix_stubs.h>

#include <rtac_base/cuda/utils.h>
#include <rtac_base/cuda/DeviceVector.h>
#include <rtac_base/cuda/HostVector.h>

#include <rtac_optix/Handle.h>
#include <rtac_optix/utils.h>
#include <rtac_optix/Context.h>
#include <rtac_optix/TraversableHandle.h>

namespace rtac { namespace optix {

class AccelerationStruct : public TraversableHandle
{
    // ABSTRACT class

    // Represent complex nodes of the rendering graph which need to be built.
    // The purpose of this graph is to hide the build operation. It must be
    // subclassed by a class implementing the sbt_width method.

    public:

    using Ptr      = Handle<AccelerationStruct>;
    using ConstPtr = Handle<const AccelerationStruct>;

    using BuildInput   = OptixBuildInput;
    using BuildOptions = OptixAccelBuildOptions;
    static BuildInput   default_build_input();
    static BuildOptions default_build_options();

    using Buffer = rtac::cuda::DeviceVector<unsigned char>;
    // This contains the CUDA stream in which the build is performed and a
    // temporaty buffer needed during the build but not used afterwards.  This
    // can be ignored by the user (defaults values will be provided). They are
    // usefull if the user wants to optimize the build process.
    struct BuildMeta { Handle<Buffer> buffer; CUstream stream; };

    protected:
    
    OptixTraversableHandle handle_;
    Context::ConstPtr      context_;
    BuildInput             buildInput_;
    BuildOptions           buildOptions_;
    Buffer                 buffer_; // contains data after build
    BuildMeta              buildMeta_;

    AccelerationStruct(const Context::ConstPtr& context,
                       const BuildInput& buildInput = default_build_input(),
                       const BuildOptions& buildOptions = default_build_options());

    public:

    virtual void build();
    
    const BuildInput& build_input() const;
    const BuildOptions& build_options() const;

    BuildInput& build_input();
    BuildOptions& build_options();

    void set_build_buffer(const Handle<Buffer>& buffer);
    void resize_build_buffer(size_t size);
    void set_build_stream(CUstream stream);
    void set_build_meta(const Handle<Buffer>& buffer, CUstream stream = 0);

    virtual operator OptixTraversableHandle();

    CUdeviceptr data();
    virtual unsigned int sbt_width() const = 0;
};

}; //namespace optix
}; //namespace rtac

#endif //_RTAC_OPTIX_TRAVERSABLE_HANDLE_H_
